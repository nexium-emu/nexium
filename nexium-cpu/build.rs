fn main() {
    println!("cargo:rustc-check-cfg=cfg(nce_runtime)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let feature = std::env::var("CARGO_FEATURE_BACKEND_NCE").is_ok();
    if feature && arch == "aarch64" && (os == "android" || os == "linux") {
        println!("cargo:rustc-cfg=nce_runtime");
    }
}

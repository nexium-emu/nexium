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
}

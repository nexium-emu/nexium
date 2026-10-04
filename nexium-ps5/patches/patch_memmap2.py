import pathlib
import sys

root = pathlib.Path(sys.argv[1])
unix = root / "src/unix.rs"
LF = "\n"


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected one {old[:70]!r}, found {text.count(old)}")
    path.write_text(text.replace(old, new))


replace(unix, """        let (map_len, map_offset) = Self::adjust_mmap_params(len, alignment as usize)?;

""", """        let (map_len, map_offset) = Self::adjust_mmap_params(len, alignment as usize)?;

        #[cfg(target_vendor = "sony")]
        if file >= 0 && prot == libc::PROT_READ {
            let ptr = unsafe { nexium_ps5_file_map(file, aligned_offset, map_len) };
            if ptr.is_null() {
                return Err(io::Error::last_os_error());
            }
            return Ok(unsafe { Self::from_raw_parts(ptr, len, map_offset) });
        }

""")
replace(unix, """        let (ptr, len, _) = self.as_mmap_params();

""", """        let (ptr, len, _) = self.as_mmap_params();

        #[cfg(target_vendor = "sony")]
        if unsafe { nexium_ps5_file_unmap(ptr, len) } == 0 {
            return;
        }

""")
replace(unix, "fn page_size() -> usize {", """#[cfg(target_vendor = "sony")]
extern "C" {
    fn nexium_ps5_file_map(fd: libc::c_int, offset: u64, len: usize) -> *mut libc::c_void;
    fn nexium_ps5_file_unmap(ptr: *mut libc::c_void, len: usize) -> libc::c_int;
}

fn page_size() -> usize {""")
print("patched", root)

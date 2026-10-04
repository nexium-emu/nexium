#!/usr/bin/env bash
# patch-rust-src.sh: the console's libc uses the FreeBSD 11 ABI (32-bit ino_t,
# FreeBSD 11 struct stat and dirent). std no longer builds against libc's
# freebsd11 configuration because of one d_fileno width; widen it in the pinned
# toolchain's rust-src, which only the PS5 build compiles.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
sysroot=$(rustc "+$RUST_TOOLCHAIN" --print sysroot)
file="$sysroot/lib/rustlib/src/rust/library/std/src/sys/fs/unix.rs"
[[ -f $file ]] || { echo "missing $file (rustup component add rust-src)" >&2; exit 1; }
if grep -q "d_ino: (\*entry_ptr).d_fileno as u64," "$file"; then
    exit 0
fi
count=$(grep -c "d_ino: (\*entry_ptr).d_fileno," "$file" || true)
[[ $count == 1 ]] || { echo "$file: expected one d_fileno site, found $count" >&2; exit 1; }
sed -i 's/d_ino: (\*entry_ptr).d_fileno,/d_ino: (*entry_ptr).d_fileno as u64,/' "$file"
echo "patched $file"

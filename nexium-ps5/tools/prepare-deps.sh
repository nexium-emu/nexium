#!/usr/bin/env bash
# prepare-deps.sh: fetch the pinned crates the PS5 build patches and apply the
# PS5 patches to them under $PS5_DEPS (never the cargo registry used by the
# desktop build).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

fetch_crate() {
    local name=$1 version=$2 checksum=$3
    local crate="$PS5_DEPS/$name-$version.crate"
    if [[ ! -f $crate ]] || [[ $(sha256sum "$crate" | cut -d' ' -f1) != "$checksum" ]]; then
        curl -sSfL -o "$crate" "https://static.crates.io/crates/$name/$name-$version.crate"
    fi
    [[ $(sha256sum "$crate" | cut -d' ' -f1) == "$checksum" ]] || { echo "$crate: checksum mismatch" >&2; exit 1; }
    rm -rf "$PS5_DEPS/$name-$version"
    tar xzf "$crate" -C "$PS5_DEPS"
}

mkdir -p "$PS5_DEPS"
fetch_crate dynarmic-sys-mythrax 0.3.5 d0d11c2cc79c1d53decb3f56f2c7bcc9ab9d2b159a134f75bd9fd83a2e771f95
python3 "$ps5_dir/patches/patch_dynarmic.py" "$PS5_DEPS/dynarmic-sys-mythrax-0.3.5" "$ps5_dir/patches/ps5_page_table.inc"
fetch_crate memmap2 0.9.11 d1219ed1b7f229ee7104d281dd01d6802fe28bb6e95d292942c4daacdeb798c0
python3 "$ps5_dir/patches/patch_memmap2.py" "$PS5_DEPS/memmap2-0.9.11"
stamp=$(cat "$ps5_dir"/patches/*.py "$ps5_dir"/patches/*.inc | sha256sum | cut -d' ' -f1)
echo "$stamp" > "$PS5_DEPS/.ps5-patch-stamp"
echo "prepared $PS5_DEPS"

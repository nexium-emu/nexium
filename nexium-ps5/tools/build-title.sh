#!/usr/bin/env bash
# build-title.sh [build-rust.sh options]
set -euo pipefail
here=$(dirname -- "${BASH_SOURCE[0]}")
bash "$here/build-rust.sh" --title "$@" > "${BUILD_LOG:-/dev/stderr}"
source "$here/env.sh"
lib="$NEXIUM_PS5_TARGET_DIR/$target_name/release/libnexium_ps5.a"
bash "$here/link-title.sh" "$lib"

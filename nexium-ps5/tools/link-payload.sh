#!/usr/bin/env bash
# link-payload.sh STATICLIB OUT.elf
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
lib=$1 out=$2
mkdir -p "$(dirname "$out")"
"$PS5_SDK/bin/prospero-clang" -o "$out" -Wl,--gc-sections "$lib" -lkernel_sys -lSceSystemService
echo "$out"

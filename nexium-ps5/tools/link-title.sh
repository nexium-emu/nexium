#!/usr/bin/env bash
# link-title.sh STATICLIB [EXTRA_ARCHIVE...]
# Links the Rust static library into the PS5 title with PS5_Vulkan's RADV link
# recipe, converts and signs eboot.bin, and assembles $NEXIUM_PS5_OUT/dist/<TITLE_ID>.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
lib=$1
shift
extra=("$@")

vulkan=$PS5_VULKAN_DIR
sdk=$PS5_SDK
archive=${RADV_ARCHIVE:-$vulkan/.deps/native/radv-release/lib/libvulkan_radeon.ps5.a}
tool="$vulkan/build/host/ps5-native-tool"
native="$vulkan/tooling/native"
param="$ps5_dir/sce_sys/param.json"
work="$NEXIUM_PS5_OUT/link"
for file in "$lib" "$archive" "$tool" "$param" "$sdk/bin/prospero-lld" "$vulkan/tools/radv-link.sh" \
        "$vulkan/runtime/libc.prx" "$repo_root/branding/png/logo-512.png"; do
    [[ -e $file ]] || { echo "missing: $file" >&2; exit 2; }
done
mkdir -p "$work/obj" "$work/stubs"
cc() { PS5_PAYLOAD_SDK="$sdk" sh "$vulkan/tooling/prospero-clang18" "$@"; }

cc -std=c++20 -O2 -fno-exceptions -fno-rtti -c "$native/app_crt.cpp" -o "$work/obj/app_crt.o"
stub() {
    local library=$1 source=$2
    cc -std=c11 -O2 -fPIC -c "$vulkan/$source" -o "$work/obj/${library}_stub.o"
    "$sdk/bin/prospero-lld" --shared -soname "${library}.prx" \
        -o "$work/stubs/${library}.so" "$work/obj/${library}_stub.o"
}
stub libSceAgc vendor/ps5/sdk/stubs/agc_canary_link_stub.c
stub libSceAgcDriver vendor/ps5/sdk/stubs/agc_driver_canary_link_stub.c

source "$vulkan/tools/radv-link.sh"
radv_link_recipe "$vulkan" "$sdk" "$archive" || exit 2
if "$sdk/bin/llvm-nm" --defined-only "$sdk/target/lib/libps5platform.a" 2>/dev/null | grep -q " T ps5_localeconv$" &&
        [[ " ${radv_link_flags[*]} " != *" --defsym=localeconv=ps5_localeconv "* ]]; then
    printf '{\n    local:\n        localeconv;\n};\n' > "$work/localeconv-local.map"
    radv_link_flags+=(--defsym=localeconv=ps5_localeconv --version-script "$work/localeconv-local.map")
fi
radv_link_flags+=(--wrap=open --wrap=close --wrap=clock_gettime)
"$sdk/bin/prospero-lld" "${radv_linker_script[@]}" --eh-frame-hdr "${radv_link_flags[@]}" \
    --version-script "$native/app-symbols.map" --exclude-libs=ALL \
    -e _start -o "$work/llvm-pie.elf" \
    "$work/obj/app_crt.o" --whole-archive "$lib" --no-whole-archive "${extra[@]}" \
    "$work/stubs/libSceAgc.so" "$work/stubs/libSceAgcDriver.so" \
    "${radv_link_inputs[@]}" \
    --as-needed "$sdk"/target/lib/*.so
"$tool" link --in "$work/llvm-pie.elf" --out "$work/eboot.elf.new" \
    --stub-dir "$sdk/target/lib" --stub "$work/stubs/libSceAgc.so" \
    --stub "$work/stubs/libSceAgcDriver.so" --module-sdk 0x02000009 \
    --companion-sdk 0x08050001 --file-name eboot.elf

title_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["titleId"])' "$param")
app="$NEXIUM_PS5_OUT/dist/$title_id"
mkdir -p "$app/sce_sys" "$app/sce_module"
"$tool" self --sign --in "$work/eboot.elf.new" --out "$app/eboot.bin" --magic 0x1D3D154F
"$tool" self --inspect --file "$app/eboot.bin" > /dev/null
cp "$param" "$app/sce_sys/param.json"
python3 - "$repo_root/branding/png/logo-512.png" "$app/sce_sys/icon0.png" <<'PY'
import sys
from PIL import Image
src = Image.open(sys.argv[1]).convert("RGBA").resize((512, 512))
bg = Image.new("RGBA", src.size, (16, 18, 28, 255))
bg.alpha_composite(src)
bg.convert("RGB").save(sys.argv[2])
PY
(cd "$vulkan/runtime" && sha256sum --check --strict --quiet libc.prx.sha256)
cp "$vulkan/runtime/libc.prx" "$app/sce_module/libc.prx"
mv "$work/eboot.elf.new" "$work/eboot.elf"
printf '==> %s: %s (eboot.bin %s bytes)\n' "$title_id" "$app" "$(stat -c %s "$app/eboot.bin")"

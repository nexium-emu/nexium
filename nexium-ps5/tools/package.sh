#!/usr/bin/env bash
# package.sh [OUT_DIR]: build the title and write an install folder (PPSA99640/) and a zip of it.
# Install: upload the PPSA99640 folder to /data/homebrew/ (PS5Upload or FTP); ShadowMountPlus registers it.
set -euo pipefail
here=$(dirname -- "${BASH_SOURCE[0]}")
source "$here/env.sh"
out=${1:-$repo_root/target/ps5-dist}
BUILD_LOG=${BUILD_LOG:-$NEXIUM_PS5_OUT/package-build.log} bash "$here/build-title.sh"
title_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["titleId"])' "$ps5_dir/sce_sys/param.json")
src="$NEXIUM_PS5_OUT/dist/$title_id"
stamp=$(date +%Y%m%d)
stage="$out/$title_id"
rm -rf "$stage"
mkdir -p "$stage/data/games"
cp -r "$src/eboot.bin" "$src/sce_sys" "$src/sce_module" "$stage/"
cat > "$stage/data/games/PUT_GAMES_HERE.txt" << 'EOF'
Copy decrypted Switch games here (.nsp .xci .dnsp .dxci .nro).
With no launch.txt, NeXium boots the first one in alphabetical order.
EOF
cat > "$stage/data/launch.txt.example" << 'EOF'
# Rename to launch.txt to pick a game and options.
game=/app0/data/games/your-game.dnsp
docked=1
volume=1.0
log=info
EOF
cp "$NEXIUM_PS5_OUT/link/llvm-pie.elf" "$out/$title_id-llvm-pie-$stamp.elf"
zip="$out/NeXium-$title_id-$stamp.zip"
rm -f "$zip"
python3 - "$out" "$title_id" "$zip" << 'PY'
import os, sys, zipfile
out, title, dest = sys.argv[1:4]
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(os.path.join(out, title)):
        for name in files:
            path = os.path.join(root, name)
            z.write(path, os.path.relpath(path, out))
PY
(cd "$out" && sha256sum "$(basename "$zip")" "$title_id/eboot.bin" "$title_id/sce_module/libc.prx")
echo "install folder: $stage"
echo "zip: $zip"

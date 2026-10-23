#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

asset="${APPIMAGE_ASSET:-nexium-linux-x86_64.AppImage}"
runtime_url="${APPIMAGE_RUNTIME_URL:-https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64}"
runtime_sha256="${APPIMAGE_RUNTIME_SHA256:?APPIMAGE_RUNTIME_SHA256 must be set}"
max_glibc="${APPIMAGE_MAX_GLIBC:-2.35}"
work="${repo_root}/target/appimage"
appdir="${work}/AppDir"
payload="${appdir}/usr/bin"

export CARGO_HOME="${CARGO_HOME:-/opt/rust/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rust/rustup}"
export PATH="${CARGO_HOME}/bin:${PATH}"
git config --global --add safe.directory '*' >/dev/null 2>&1 || true

if ! command -v cargo >/dev/null 2>&1; then
    curl -fsSL --retry 3 https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path
fi
rustup update --no-self-update stable
rustc --version
cargo --version
cmake --version | head -n 1
"${CXX:-c++}" --version | head -n 1
ldd --version | head -n 1

cargo build --release -p nexium-gui --bin nexium
unset RUNNER_TEMP
bash .github/scripts/build-ffmpeg.sh
ffmpeg_dist="${repo_root}/target/ffmpeg-dist"

rm -rf "${work}"
mkdir -p "${payload}" "${appdir}/usr/share/applications" "${appdir}/usr/share/icons/hicolor/256x256/apps"
install -m 0755 target/release/nexium "${payload}/nexium"
strip --strip-debug "${payload}/nexium"
cp -a "${ffmpeg_dist}/." "${payload}/"
cp LICENSE README.md "${payload}/"
install -m 0755 packaging/appimage/AppRun "${appdir}/AppRun"
install -m 0644 packaging/appimage/nexium.desktop "${appdir}/nexium.desktop"
install -m 0644 packaging/appimage/nexium.desktop "${appdir}/usr/share/applications/nexium.desktop"
install -m 0644 branding/png/logo-256.png "${appdir}/nexium.png"
install -m 0644 branding/png/logo-256.png "${appdir}/usr/share/icons/hicolor/256x256/apps/nexium.png"
ln -sf nexium.png "${appdir}/.DirIcon"

for f in ffmpeg THIRD_PARTY_NOTICES.txt licenses/FFmpeg/LICENSE.md licenses/FFmpeg/COPYING.LGPLv2.1 \
    licenses/FFmpeg/BUILD.txt "licenses/FFmpeg/ffmpeg-${FFMPEG_VERSION:-9.0.1}.tar.xz" \
    "licenses/FFmpeg/ffmpeg-${FFMPEG_VERSION:-9.0.1}.tar.xz.asc" licenses/FFmpeg/ffmpeg-devel.asc \
    licenses/FFmpeg/CHANGES.diff; do
    test -f "${payload}/${f}" || { echo "::error::AppImage payload is missing ${f}"; exit 1; }
done
test -x "${payload}/ffmpeg"

for bin in nexium ffmpeg; do
    needed="$(objdump -T "${payload}/${bin}" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -n 1)"
    newest="$(printf '%s\n%s\n' "${needed}" "${max_glibc}" | sort -V | tail -n 1)"
    echo "${bin}: needs glibc ${needed}, libstdc++ $(objdump -T "${payload}/${bin}" | grep -o 'GLIBCXX_[0-9.]*' | sort -V | tail -n 1 || true)"
    if [[ "${newest}" != "${max_glibc}" ]]; then
        echo "::error::${bin} requires glibc ${needed}, newer than the ${max_glibc} the AppImage promises"
        exit 1
    fi
done
echo "nexium shared libraries:"
ldd "${payload}/nexium" | sort

runtime="${work}/runtime-x86_64"
curl -fsSL --retry 3 -o "${runtime}" "${runtime_url}"
echo "${runtime_sha256}  ${runtime}" | sha256sum -c -

mksquashfs "${appdir}" "${work}/nexium.squashfs" -root-owned -noappend -no-xattrs \
    -comp zstd -Xcompression-level 19 -b 1M -quiet
mkdir -p dist
cat "${runtime}" "${work}/nexium.squashfs" > "dist/${asset}"
chmod 0755 "dist/${asset}"

(
    cd "${work}"
    rm -rf squashfs-root
    "${repo_root}/dist/${asset}" --appimage-extract >/dev/null
    test -x squashfs-root/AppRun
    test -x squashfs-root/usr/bin/nexium
    test -x squashfs-root/usr/bin/ffmpeg
    test -f squashfs-root/nexium.desktop
    cmp squashfs-root/usr/bin/nexium "${payload}/nexium"
)
ls -l "dist/${asset}"

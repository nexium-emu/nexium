#!/usr/bin/env bash
set -euo pipefail

ffmpeg_version="${FFMPEG_VERSION:-9.0.1}"
ffmpeg_sha256="${FFMPEG_SOURCE_SHA256:-cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635}"
signing_fingerprint="${FFMPEG_SIGNING_FINGERPRINT:-FCF986EA15E6E293A5644F10B4322F04D67658D8}"
source_url="https://ffmpeg.org/releases/ffmpeg-${ffmpeg_version}.tar.xz"
signature_url="${source_url}.asc"
signing_key_url="https://ffmpeg.org/ffmpeg-devel.asc"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner_temp=""
if [[ -n "${RUNNER_TEMP:-}" ]]; then
    runner_temp="${RUNNER_TEMP}"
    if [[ "${RUNNER_OS:-}" == "Windows" ]]; then
        runner_temp="$(cygpath -u "${runner_temp}")"
    fi
    work_root="${runner_temp}/nexium-ffmpeg-build"
    dist_root="${runner_temp}/nexium-ffmpeg-dist"
else
    work_root="${repo_root}/target/ffmpeg-build"
    dist_root="${repo_root}/target/ffmpeg-dist"
fi
source_dir="${work_root}/ffmpeg-${ffmpeg_version}"
archive="${work_root}/ffmpeg-${ffmpeg_version}.tar.xz"
signature="${archive}.asc"
signing_key="${work_root}/ffmpeg-devel.asc"
keyring="${work_root}/gnupg"

if [[ -n "${runner_temp}" ]]; then
    rm -rf "${repo_root}/target/ffmpeg-build" "${repo_root}/target/ffmpeg-dist"
fi

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

mkdir -p "${work_root}"
if [[ ! -f "${archive}" ]] || [[ "$(hash_file "${archive}")" != "${ffmpeg_sha256}" ]]; then
    rm -f "${archive}"
    curl --fail --location --retry 3 --output "${archive}" "${source_url}"
fi

actual_sha256="$(hash_file "${archive}")"
if [[ "${actual_sha256}" != "${ffmpeg_sha256}" ]]; then
    echo "FFmpeg source checksum mismatch: ${actual_sha256}" >&2
    exit 1
fi

curl --fail --location --retry 3 --output "${signature}" "${signature_url}"
curl --fail --location --retry 3 --output "${signing_key}" "${signing_key_url}"
rm -rf "${keyring}"
mkdir -p "${keyring}"
if [[ "${RUNNER_OS:-}" != "Windows" ]]; then
    chmod 700 "${keyring}"
fi
actual_fingerprint="$(gpg --batch --homedir "${keyring}" --show-keys --with-colons "${signing_key}" | awk -F: '$1 == "fpr" { print $10; exit }')"
if [[ "${actual_fingerprint}" != "${signing_fingerprint}" ]]; then
    echo "FFmpeg signing key mismatch: ${actual_fingerprint}" >&2
    exit 1
fi
gpg --batch --quiet --homedir "${keyring}" --import "${signing_key}"
gpg --batch --homedir "${keyring}" --verify "${signature}" "${archive}"
rm -rf "${source_dir}" "${dist_root}"
tar -xf "${archive}" -C "${work_root}"

configure_args=(
    --disable-autodetect
    --disable-doc
    --disable-debug
    --disable-network
    --disable-everything
    --disable-gpl
    --disable-nonfree
    --disable-version3
    --disable-shared
    --enable-static
    --enable-ffmpeg
    --disable-ffplay
    --disable-ffprobe
    --enable-protocol=pipe
    --enable-demuxer=h264
    --enable-demuxer=ivf
    --enable-parser=h264
    --enable-parser=vp9
    --enable-decoder=h264
    --enable-decoder=vp9
    --enable-encoder=rawvideo
    --enable-muxer=rawvideo
    --enable-filter=format
    --enable-filter=scale
    --enable-swscale
)

binary_name="ffmpeg"
if [[ "${RUNNER_OS:-}" == "Windows" ]]; then
    msvc_cl="$(command -v cl.exe || true)"
    if [[ -z "${msvc_cl}" ]]; then
        echo "MSVC cl.exe is not available" >&2
        exit 1
    fi
    export PATH="$(dirname "${msvc_cl}"):${PATH}"
    configure_args+=(--toolchain=msvc --arch=x86_64)
    binary_name="ffmpeg.exe"
fi
if [[ "$(uname -m)" == "x86_64" ]] && ! command -v nasm >/dev/null 2>&1; then
    configure_args+=(--disable-x86asm)
fi

pushd "${source_dir}" >/dev/null
./configure "${configure_args[@]}"
jobs="$(command -v nproc >/dev/null 2>&1 && nproc || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
make -j "${jobs}" "${binary_name}"
popd >/dev/null

license_root="${dist_root}/licenses/FFmpeg"
mkdir -p "${license_root}"
cp "${source_dir}/${binary_name}" "${dist_root}/${binary_name}"
chmod 755 "${dist_root}/${binary_name}"
cp "${repo_root}/THIRD_PARTY_NOTICES.txt" "${dist_root}/THIRD_PARTY_NOTICES.txt"
cp "${source_dir}/LICENSE.md" "${license_root}/LICENSE.md"
cp "${source_dir}/COPYING.LGPLv2.1" "${license_root}/COPYING.LGPLv2.1"
cp "${archive}" "${license_root}/ffmpeg-${ffmpeg_version}.tar.xz"
cp "${signature}" "${license_root}/ffmpeg-${ffmpeg_version}.tar.xz.asc"
cp "${signing_key}" "${license_root}/ffmpeg-devel.asc"
: > "${license_root}/CHANGES.diff"

buildconf="$(${dist_root}/${binary_name} -buildconf 2>&1)"
if grep -Fq -- '--enable-gpl' <<<"${buildconf}"; then
    echo "FFmpeg unexpectedly enabled GPL components" >&2
    exit 1
fi
if grep -Fq -- '--enable-nonfree' <<<"${buildconf}"; then
    echo "FFmpeg unexpectedly enabled nonfree components" >&2
    exit 1
fi
if grep -Fq -- '--enable-version3' <<<"${buildconf}"; then
    echo "FFmpeg unexpectedly enabled version 3 components" >&2
    exit 1
fi
grep -Fq -- '--disable-gpl' <<<"${buildconf}"
grep -Fq -- '--disable-nonfree' <<<"${buildconf}"
grep -Fq -- '--disable-version3' <<<"${buildconf}"

require_component() {
    local listing="$1"
    local component="$2"
    "${dist_root}/${binary_name}" -hide_banner "${listing}" 2>&1 \
        | grep -Eq "[[:space:]]${component}([[:space:]]|$)"
}

require_component -decoders h264
require_component -decoders vp9
require_component -demuxers h264
require_component -demuxers ivf
require_component -encoders rawvideo
require_component -muxers rawvideo
require_component -protocols pipe
require_component -filters format
require_component -filters scale
binary_sha256="$(hash_file "${dist_root}/${binary_name}")"

{
    echo "FFmpeg ${ffmpeg_version}"
    echo
    echo "Build platform: ${RUNNER_OS:-$(uname -s)} $(uname -m)"
    echo "Binary SHA-256: ${binary_sha256}"
    echo "Source: ${source_url}"
    echo "Source SHA-256: ${ffmpeg_sha256}"
    echo "Source signature: ${signature_url}"
    echo "Signing key: ${signing_key_url}"
    echo "Signing fingerprint: ${signing_fingerprint}"
    echo "Source modifications: none"
    echo "Build recipe: https://github.com/nexium-emu/nexium/blob/${GITHUB_SHA:-main}/.github/scripts/build-ffmpeg.sh"
    echo
    printf 'Configure arguments:'
    printf ' %q' "${configure_args[@]}"
    echo
    echo
    "${dist_root}/${binary_name}" -version
} > "${license_root}/BUILD.txt"

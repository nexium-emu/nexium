#!/usr/bin/env bash
# setup-stack.sh: install the PS5 build stack at pinned revisions.
# Host: Ubuntu 26.04 (WSL2 on the Windows PC). Run as a user that may apt-get (root in WSL).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

declare -A PIN=(
    [PS5_Vulkan]=3f3ee69607013b345d2baa6d6a37c86745649a08
    [PS5_Mesa]=0b2d6d1a61d9bbf89cf8beb88a696144f67c61f8
    [PS5_PayloadSDK]=fa69d00fe974a47a20009d32c9780c259a05a08f
    [PS5_VulkanTemplate]=80defeab31c2e6c752573a7a99d611f898edfbbb
)
FTPSRV_URL=https://github.com/ps5-payload-dev/ftpsrv/releases/download/v0.21.1/ftpsrv-ps5.elf
FTPSRV_SHA=7d4b31c83eae4e056580482a3a074e1db25922efc74ac1e439473b45d71a5938
KLOGSRV_URL=https://github.com/ps5-payload-dev/klogsrv/releases/download/v0.9/klogsrv-ps5.elf
KLOGSRV_SHA=e828ec144231f81547cb58bc7d2c396fa984be0c2295f31364b58017816dcceb

if [[ ${SKIP_APT:-0} != 1 ]]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq clang lld llvm cmake ninja-build meson python3-numpy python3-pil python3-mako \
        python3-yaml python3-packaging python3-ply glslang-tools rsync zstd make bison flex pkg-config git curl \
        unzip xz-utils file netcat-openbsd libllvmspirvlib-21-dev libclc-21-dev libclang-21-dev llvm-21-dev \
        clang-21 libpolly-21-dev llvm-spirv-21 spirv-tools spirv-tools-dev spirv-tools-headers spirv-headers \
        libelf-dev libboost-dev nasm
fi

command -v rustup > /dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
source "$HOME/.cargo/env"
rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal -c rust-src

mkdir -p "$PS5_STACK"
for repo in "${!PIN[@]}"; do
    dir="$PS5_STACK/$repo"
    [[ -d $dir/.git ]] || git clone -q "https://github.com/mihawk-99/$repo.git" "$dir"
    git -C "$dir" fetch -q origin
    git -C "$dir" checkout -q "${PIN[$repo]}"
    echo "$repo at $(git -C "$dir" rev-parse --short HEAD)"
done

(
    cd "$PS5_VULKAN_DIR"
    bash tools/setup-native-dependencies.sh
    make > build-make.log 2>&1 || true
    [[ -x build/host/ps5-native-tool && -f runtime/libc.prx ]] || { echo "native tool or libc.prx missing: build-make.log" >&2; exit 1; }
    (cd runtime && sha256sum --check --strict libc.prx.sha256)
    bash tools/build-radv.sh release
    PS5_PAYLOAD_SDK="$PS5_SDK" bash tools/build-ps5vkctl.sh
)

payloads="$PS5_STACK/../payloads"
mkdir -p "$payloads"
fetch() {
    local url=$1 sha=$2 out=$3
    [[ -f $out && $(sha256sum "$out" | cut -d' ' -f1) == "$sha" ]] || curl -sSfL -o "$out" "$url"
    [[ $(sha256sum "$out" | cut -d' ' -f1) == "$sha" ]] || { echo "$out: checksum mismatch" >&2; exit 1; }
}
fetch "$FTPSRV_URL" "$FTPSRV_SHA" "$payloads/ftpsrv-ps5-v0.21.1.elf"
fetch "$KLOGSRV_URL" "$KLOGSRV_SHA" "$payloads/klogsrv-ps5-v0.9.elf"
bash "$ps5_dir/tools/prepare-deps.sh"
echo "stack ready in $PS5_STACK"

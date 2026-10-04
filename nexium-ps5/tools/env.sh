#!/usr/bin/env bash
ps5_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(dirname "$ps5_dir")
PS5_STACK=${PS5_STACK:-/opt/ps5/stack}
PS5_VULKAN_DIR=${PS5_VULKAN_DIR:-$PS5_STACK/PS5_Vulkan}
PS5_SDK=${PS5_SDK:-$PS5_VULKAN_DIR/.deps/native/ps5-payload-sdk}
NEXIUM_PS5_TARGET_DIR=${NEXIUM_PS5_TARGET_DIR:-/opt/ps5/nexium-target}
NEXIUM_PS5_OUT=${NEXIUM_PS5_OUT:-/opt/ps5/nexium-out}
PS5_DEPS=${PS5_DEPS:-/opt/ps5/deps}
RUST_TOOLCHAIN=$(sed -n 's/^channel = "\(.*\)"/\1/p' "$ps5_dir/rust-toolchain.toml")
target_name=x86_64-nexium-ps5
target_spec=$ps5_dir/$target_name.json
[[ -f $HOME/.cargo/env ]] && source "$HOME/.cargo/env"
if [[ -f $PS5_VULKAN_DIR/.env ]]; then
    set -a
    source "$PS5_VULKAN_DIR/.env"
    set +a
fi
PS5_HOST=${PS5_HOST:-}
ELFLDR_PORT=${ELFLDR_PORT:-9021}
export PS5_PAYLOAD_SDK=$PS5_SDK
export PS5_CLANG=${PS5_CLANG:-$(command -v clang || true)}

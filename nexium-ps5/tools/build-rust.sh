#!/usr/bin/env bash
# build-rust.sh [--title] [--debug] [--features LIST]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

profile=release
features=()
while (( $# )); do
    case $1 in
        --title) features+=(title) ;;
        --debug) profile=dev ;;
        --features) features+=("$2"); shift ;;
        *) echo "usage: ${0##*/} [--title] [--debug] [--features LIST]" >&2; exit 2 ;;
    esac
    shift
done

export RUSTFLAGS="--cfg libc_unstable_freebsd_version=\"11\" -C force-frame-pointers=yes ${RUSTFLAGS:-}"
export CARGO_TARGET_DIR=$NEXIUM_PS5_TARGET_DIR
export CC_x86_64_nexium_ps5="$PS5_SDK/bin/prospero-clang"
export CXX_x86_64_nexium_ps5="$PS5_SDK/bin/prospero-clang++"
export AR_x86_64_nexium_ps5="$PS5_SDK/bin/prospero-ar"
export CRATE_CC_NO_DEFAULTS=1
boost_include="$PS5_DEPS/boost-include"
if [[ ! -e $boost_include/boost ]]; then
    mkdir -p "$boost_include"
    ln -sfn /usr/include/boost "$boost_include/boost"
fi
export BOOST_ROOT=${BOOST_ROOT:-$boost_include}
export CFLAGS_x86_64_nexium_ps5="-fPIC -fno-omit-frame-pointer -ffunction-sections -fdata-sections -fdenormal-fp-math=ieee"
export CXXFLAGS_x86_64_nexium_ps5="$CFLAGS_x86_64_nexium_ps5 -DFMT_CONSTEVAL="
export CMAKE_TOOLCHAIN_FILE_x86_64_nexium_ps5="$ps5_dir/tools/ps5-toolchain.cmake"
export CMAKE_GENERATOR=Ninja

bash "$ps5_dir/tools/patch-rust-src.sh"
dynarmic="$PS5_DEPS/dynarmic-sys-mythrax-0.3.5"
stamp=$(cat "$ps5_dir"/patches/*.py "$ps5_dir"/patches/*.inc | sha256sum | cut -d' ' -f1)
if [[ ! -f $PS5_DEPS/.ps5-patch-stamp || $(< "$PS5_DEPS/.ps5-patch-stamp") != "$stamp" ]]; then
    bash "$ps5_dir/tools/prepare-deps.sh"
fi
feature_args=()
(( ${#features[@]} )) && feature_args=(--features "$(IFS=,; echo "${features[*]}")")

cd "$ps5_dir"
cargo "+$RUST_TOOLCHAIN" build -Zbuild-std=std,panic_abort -Zjson-target-spec --target "$target_spec" \
    --config "resolver.lockfile-path=\"$ps5_dir/lock/Cargo.lock\"" \
    --config "patch.crates-io.dynarmic-sys-mythrax.path=\"$dynarmic\"" \
    --config "patch.crates-io.memmap2.path=\"$PS5_DEPS/memmap2-0.9.11\"" \
    -p nexium-ps5 --profile "$profile" "${feature_args[@]}"
dir=$([[ $profile == dev ]] && echo debug || echo release)
lib="$CARGO_TARGET_DIR/$target_name/$dir/libnexium_ps5.a"
[[ -f $lib ]] || { echo "missing $lib" >&2; exit 1; }
renames=()
while read -r symbol; do
    renames+=(--redefine-sym "$symbol=${symbol%%@*}")
done < <("$PS5_SDK/bin/llvm-nm" -u "$lib" 2>/dev/null | awk '$2 ~ /@FBSD_/ {print $2}' | sort -u)
if (( ${#renames[@]} )); then
    "$PS5_SDK/bin/llvm-objcopy" "${renames[@]}" "$lib"
fi
echo "$lib"

#!/usr/bin/env bash
# run-title.sh [--no-deploy] [--timeout S] [--until REGEX]
# Uploads $NEXIUM_PS5_OUT/dist/<TITLE_ID> over the console's FTP server, launches
# it through ps5vkctl, captures klog until the end line, a crash or the title's
# exit, then closes it if it is still running.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
deploy=true
timeout=180
until_re='\[nexium-ps5\].* exit failures='
while (( $# )); do
    case $1 in
        --no-deploy) deploy=false ;;
        --timeout) timeout=$2; shift ;;
        --until) until_re=$2; shift ;;
        *) echo "usage: ${0##*/} [--no-deploy] [--timeout S] [--until REGEX]" >&2; exit 2 ;;
    esac
    shift
done
title_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["titleId"])' "$ps5_dir/sce_sys/param.json")
app="$NEXIUM_PS5_OUT/dist/$title_id"
[[ -d $app ]] || { echo "no $app: run tools/build-title.sh first" >&2; exit 2; }
if $deploy; then
    python3 "$PS5_VULKAN_DIR/tools/deploy-title-folder.py" --always eboot.bin "$app"
fi
run_dir="$NEXIUM_PS5_OUT/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
cd "$PS5_VULKAN_DIR"
python3 tools/run-title.py "$title_id" --until "$until_re" --timeout "$timeout" \
    --echo '\[nexium-ps5\]|fatal|Fatal|signal|FAILED' \
    --elf "$NEXIUM_PS5_OUT/link/llvm-pie.elf" --output "$run_dir/klog.log" || status=$?
echo "klog: $run_dir/klog.log"
exit "${status:-0}"

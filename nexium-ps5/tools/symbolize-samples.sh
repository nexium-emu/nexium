#!/usr/bin/env bash
# symbolize-samples.sh KLOG [FILTER]
# Resolves "sample:" lines from the title's sampler (NEXIUM_PS5_SAMPLE_MS) against link/llvm-pie.elf.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
klog=$1
filter=${2:-.}
elf="$NEXIUM_PS5_OUT/link/llvm-pie.elf"
anchor_link=$("$PS5_SDK/bin/llvm-nm" "$elf" | awk '$3 == "nexium_ps5_sample_anchor" { print $1 }')
anchor_run=$(grep -a "sample: anchor" "$klog" | tail -1 | sed 's/.*anchor 0x\([0-9a-f]*\).*/\1/')
slide=$(( 0x$anchor_run - 0x$anchor_link ))
text_end=$(( 0x$("$PS5_SDK/bin/llvm-nm" "$elf" | sort | tail -1 | cut -d' ' -f1) + slide ))
grep -a "sample: " "$klog" | grep -v "sample: anchor" | grep -a -- "$filter" | while read -r line; do
    name=$(sed 's/.*sample: \(.*\) rax .*/\1/' <<< "$line")
    regs=$(sed 's/.* \(rax .* rip [0-9a-f]*\) .*/\1/' <<< "$line")
    printf '== %s %s\n' "$name" "$regs"
    for a in $(sed 's/.* rip \([0-9a-f]*\) stack \(.*\)/\1 \2/' <<< "$line"); do
        v=$(( 0x$a ))
        if (( v >= slide && v < text_end )); then printf '0x%x\n' $(( v - slide )); fi
    done | xargs -r llvm-addr2line -f -C -e "$elf" 2>/dev/null |
        awk 'NR % 2 == 1' | sed 's/::h[0-9a-f]\{16\}//' | awk '!seen[$0]++' | sed 's/^/   /' | head -8
done

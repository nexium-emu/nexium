#!/usr/bin/env bash
# console-setup.sh HOST: once per console boot, after the jailbreak's autoload has run
# (elfldr on 9021, ShadowMountPlus, kstuff). Loads ftpsrv (2121), klogsrv (3232) and
# ps5vkctl (9111) through the ELF loader and records the address for the console tools.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
host=${1:-${PS5_HOST:-}}
[[ -n $host ]] || { echo "usage: ${0##*/} HOST" >&2; exit 2; }
printf 'PS5_HOST=%s\nFTP_PORT=2121\nKLOG_PORT=3232\nPS5VKCTL_PORT=9111\n' "$host" > "$PS5_VULKAN_DIR/.env"
payloads="$PS5_STACK/../payloads"
port_open() { timeout 2 bash -c "echo > /dev/tcp/$host/$1" 2> /dev/null; }
port_open "$ELFLDR_PORT" || { echo "no ELF loader on $host:$ELFLDR_PORT: run the jailbreak first" >&2; exit 1; }
send() {
    local port=$1 elf=$2
    if port_open "$port"; then
        echo "port $port already serving; not reloading $(basename "$elf")"
        return
    fi
    timeout 5 nc -N "$host" "$ELFLDR_PORT" < "$elf" > /dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        port_open "$port" && { echo "$(basename "$elf") up on $port"; return; }
        sleep 0.5
    done
    echo "$(basename "$elf") did not open port $port" >&2
    exit 1
}
send 2121 "$payloads/ftpsrv-ps5-v0.21.1.elf"
send 3232 "$payloads/klogsrv-ps5-v0.9.elf"
send 9111 "$PS5_VULKAN_DIR/build/ps5vkctl/ps5vkctl.elf"
(cd "$PS5_VULKAN_DIR" && python3 tools/ps5_console.py payload)

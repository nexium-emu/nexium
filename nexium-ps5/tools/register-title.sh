#!/usr/bin/env bash
# register-title.sh: make ShadowMountPlus register a title folder it could not register
# ("AppInstallTitleDir bridge unavailable" in /data/shadowmount/debug.log). SMP installs its
# ShellCore hooks only at start-up and registers new /data/homebrew folders then; loading the
# console's own SMP ELF again stops the running instance cleanly and starts a fresh one.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
host=${PS5_HOST:?PS5_HOST unset: run console-setup.sh first}
smp=${SMP_ELF:-/data/pldmgr/payloads/ShadowMountPlus/ShadowMountPlus_1.7beta2.elf}
local_copy="$NEXIUM_PS5_OUT/$(basename "$smp")"
mkdir -p "$NEXIUM_PS5_OUT"
python3 - "$host" "$smp" "$local_copy" <<'PY'
import sys
from ftplib import FTP
host, remote, local = sys.argv[1:4]
f = FTP(); f.connect(host, 2121, timeout=30); f.login()
with open(local, "wb") as out:
    f.retrbinary("RETR " + remote, out.write)
f.quit()
PY
timeout 8 nc -N "$host" "$ELFLDR_PORT" < "$local_copy" 2>/dev/null | tr -d '\r' | grep -E "RESTART|Installed|Register|Found" || true
python3 - "$host" <<'PY'
import io, sys, time
from ftplib import FTP
time.sleep(4)
f = FTP(); f.connect(sys.argv[1], 2121, timeout=30); f.login()
buf = io.BytesIO(); f.retrbinary("RETR /data/shadowmount/debug.log", buf.write); f.quit()
lines = [l for l in buf.getvalue().decode("utf-8", "replace").splitlines() if "SHELLFLAG" not in l]
print("\n".join(lines[-15:]))
PY

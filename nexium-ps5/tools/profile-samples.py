#!/usr/bin/env python3
"""profile-samples.py KLOG ELF [--from S] [--to S] [--threads a,b] [--depth N]

Aggregates the title sampler's "sample:" lines (NEXIUM_PS5_SAMPLE_MS) into a flat
profile per thread: the leaf function, or the first code frame on the stack when
the leaf is in a system library, plus the most common short call chains.
"""
import argparse
import collections
import re
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("klog")
ap.add_argument("elf")
ap.add_argument("--from", dest="start", type=float, default=0)
ap.add_argument("--to", dest="end", type=float, default=1e9)
ap.add_argument("--threads", default="")
ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--inclusive", action="store_true")
ap.add_argument("--focus", default="")
ap.add_argument("--callers", default="")
a = ap.parse_args()

line_re = re.compile(r"\]\s+([0-9.]+) sample: (\S+) rax \S+ ret (\S+) rip (\S+) stack ?(.*)$")
anchor_run = None
samples = []
for raw in open(a.klog, "rb"):
    text = raw.decode("utf-8", "replace").rstrip()
    m = re.search(r"sample: anchor 0x([0-9a-f]+)", text)
    if m:
        anchor_run = int(m.group(1), 16)
        continue
    m = line_re.search(text)
    if not m:
        continue
    t = float(m.group(1))
    if not (a.start <= t <= a.end):
        continue
    name = m.group(2)
    if a.threads and not any(name.startswith(p) for p in a.threads.split(",")):
        continue
    words = [int(w, 16) for w in m.group(5).split()] if m.group(5) else []
    samples.append((name, int(m.group(4), 16), int(m.group(3), 16), words))

nm = subprocess.run(["llvm-nm", "--defined-only", a.elf], capture_output=True, text=True).stdout
anchor_link = next(int(l.split()[0], 16) for l in nm.splitlines() if l.endswith(" nexium_ps5_sample_anchor"))
text_syms = sorted(int(l.split()[0], 16) for l in nm.splitlines() if len(l.split()) == 3 and l.split()[1] in "tT")
slide = anchor_run - anchor_link
lo, hi = text_syms[0], text_syms[-1] + 0x10000

import bisect
nmc = subprocess.run(["llvm-nm", "--defined-only", "-n", "-C", a.elf], capture_output=True, text=True).stdout
sym_addrs, sym_names = [], []
for l in nmc.splitlines():
    parts = l.split(" ", 2)
    if len(parts) == 3 and parts[1] in "tTwW":
        sym_addrs.append(int(parts[0], 16))
        sym_names.append(re.sub(r"::h[0-9a-f]{16}$", "", parts[2]))
names = {}
def sym(v):
    off = v - slide
    if not (lo <= off < hi):
        return None
    if off in names:
        return names[off]
    i = bisect.bisect_right(sym_addrs, off) - 1
    n = sym_names[i] if i >= 0 else None
    names[off] = n
    return n

per_leaf = collections.defaultdict(collections.Counter)
per_incl = collections.defaultdict(collections.Counter)
focus_children = collections.defaultdict(collections.Counter)
callers = collections.defaultdict(collections.Counter)
per_chain = collections.defaultdict(collections.Counter)
totals = collections.Counter()
for name, rip, ret, words in samples:
    totals[name] += 1
    frames = []
    leaf = sym(rip)
    if leaf is None:
        leaf = "[lib %x]" % rip if rip >= 0x800000000 else "[jit/other]"
        r = sym(ret)
        if r:
            frames.append(r)
    for w in words:
        s = sym(w)
        if s and (not frames or frames[-1] != s):
            frames.append(s)
    attributed = leaf if not leaf.startswith("[") or not frames else f"{leaf} <- {frames[0]}"
    per_leaf[name][attributed[:200]] += 1
    chain = " <- ".join(f[:60] for f in ([leaf] + frames)[: a.depth])
    per_chain[name][chain] += 1
    full = [leaf] + frames
    for fn in set(full):
        per_incl[name][fn[:160]] += 1
    if a.callers:
        for i, fn in enumerate(full):
            if a.callers in fn:
                chain = " <- ".join(f[:70] for f in full[i + 1:i + 1 + a.depth])
                callers[name][chain] += 1
                break
    if a.focus:
        for i, fn in enumerate(full):
            if a.focus in fn:
                child = full[i - 1] if i > 0 else "(self)"
                focus_children[name][child[:160]] += 1
                break

for name in sorted(totals):
    print(f"## {name} ({totals[name]} samples)")
    for k, v in per_leaf[name].most_common(a.top):
        print(f"  {100 * v / totals[name]:5.1f}%  {k}")
    if a.inclusive:
        print(f"  -- inclusive")
        for k, v in per_incl[name].most_common(a.top):
            print(f"  {100 * v / totals[name]:5.1f}%  {k}")
    if a.callers and callers[name]:
        print(f"  -- callers of '{a.callers}'")
        for k, v in callers[name].most_common(a.top):
            print(f"  {100 * v / totals[name]:5.1f}%  {k}")
    if a.focus and focus_children[name]:
        print(f"  -- under '{a.focus}'")
        for k, v in focus_children[name].most_common(a.top):
            print(f"  {100 * v / totals[name]:5.1f}%  {k}")

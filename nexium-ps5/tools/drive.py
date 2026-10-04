#!/usr/bin/env python3
"""Run a game in the deployed NeXium PS5 title and drive it with a timed script.

drive.py NAME GAME_PATH [--seconds N] [--snap S] [--env K=V ...] STEP...

Steps: wait:S  press:BTN[,BTN]  hold:BTN[,BTN]:S  stick:LX,LY:S
BTN uses NeXium's inject names (A B X Y L R ZL ZR PLUS MINUS DUP DDOWN DLEFT DRIGHT LS RS).
Input goes through NEXIUM_HID_INJECT, a file in the title folder rewritten over FTP.
Results land in $NEXIUM_PS5_OUT/drive/NAME: the title log, snapshots, and a contact sheet.
"""
import argparse
import io
import os
import subprocess
import sys
import threading
import time
from ftplib import FTP

HOST = os.environ.get("PS5_HOST", "192.168.1.165")
PORT = int(os.environ.get("FTP_PORT", "2121"))
TITLE = "/data/homebrew/PPSA99640"
DATA = TITLE + "/data"
INJECT = DATA + "/inject.txt"


def ftp():
    f = FTP()
    f.connect(HOST, PORT, timeout=15)
    f.login()
    return f


def names_in(f, path):
    lines = []
    try:
        f.cwd(path)
        f.retrlines("LIST", lines.append)
    except Exception:
        return []
    return [l.split()[-1] for l in lines if not l.startswith("d")]


def delete(f, path):
    try:
        f.sendcmd("DELE " + path)
    except Exception:
        pass


def put(f, path, text):
    f.storbinary("STOR " + path, io.BytesIO(text.encode()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("game")
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--snap", type=float, default=3.0)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--log", default="info")
    ap.add_argument("steps", nargs="*")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(os.environ.get("NEXIUM_PS5_OUT", "/opt/ps5/nexium-out"), "drive", a.name)
    os.makedirs(out, exist_ok=True)

    f = ftp()
    for n in [n for n in names_in(f, DATA + "/snapshots") if n.endswith(".bmp")]:
        delete(f, DATA + "/snapshots/" + n)
    cfg = [f"game={a.game}", "docked=1", f"exit_after={a.seconds}", f"snapshot_every={a.snap}",
           f"snapshot_limit={int(a.seconds / a.snap) + 2}", f"log={a.log}", f"env.NEXIUM_HID_INJECT=/app0/data/inject.txt"]
    cfg += [f"env.{e}" for e in a.env]
    put(f, DATA + "/launch.txt", "\n".join(cfg) + "\n")
    put(f, INJECT, "")
    f.quit()

    run = subprocess.Popen(["bash", os.path.join(here, "run-title.sh"), "--no-deploy", "--timeout", str(a.seconds + 60),
                            "--until", "exit failures="], stdout=open(os.path.join(out, "run.out"), "w"),
                           stderr=subprocess.STDOUT)
    t0 = time.time()
    log = open(os.path.join(out, "steps.txt"), "w")
    f = ftp()

    def note(text):
        line = f"{time.time() - t0:7.2f} {text}"
        log.write(line + "\n")
        log.flush()
        print(line, flush=True)

    for step in a.steps:
        kind, _, rest = step.partition(":")
        if run.poll() is not None:
            note("title exited; stopping script")
            break
        if kind == "wait":
            time.sleep(float(rest))
        elif kind == "press":
            put(f, INJECT, f"buttons={rest}")
            note(f"press {rest}")
            time.sleep(0.2)
            put(f, INJECT, "")
            time.sleep(0.2)
        elif kind == "hold":
            btn, secs = rest.rsplit(":", 1)
            put(f, INJECT, f"buttons={btn}")
            note(f"hold {btn} {secs}s")
            time.sleep(float(secs))
            put(f, INJECT, "")
        elif kind == "stick":
            xy, secs = rest.rsplit(":", 1)
            lx, ly = xy.split(",")
            put(f, INJECT, f"lx={lx} ly={ly}")
            note(f"stick {lx},{ly} {secs}s")
            time.sleep(float(secs))
            put(f, INJECT, "")
        else:
            note(f"unknown step {step}")
    try:
        put(f, INJECT, "")
        f.quit()
    except Exception:
        pass
    run.wait()
    note(f"run finished rc={run.returncode}")

    f = ftp()
    with open(os.path.join(out, "nexium-ps5.log"), "wb") as o:
        f.retrbinary("RETR " + DATA + "/nexium-ps5.log", o.write)
    shots = sorted(n for n in names_in(f, DATA + "/snapshots") if n.endswith(".bmp"))
    for n in shots:
        with open(os.path.join(out, n), "wb") as o:
            f.retrbinary("RETR " + DATA + "/snapshots/" + n, o.write)
    delete(f, DATA + "/launch.txt")
    f.quit()
    try:
        from PIL import Image
        imgs = [Image.open(os.path.join(out, n)) for n in shots]
        if imgs:
            w, h = 320, 180
            cols = 4
            rows = (len(imgs) + cols - 1) // cols
            sheet = Image.new("RGB", (w * cols, h * rows))
            for i, im in enumerate(imgs):
                sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
            sheet.save(os.path.join(out, "sheet.png"))
    except Exception as e:
        note(f"contact sheet failed: {e}")
    note(f"{len(shots)} snapshots in {out}")


if __name__ == "__main__":
    sys.exit(main())

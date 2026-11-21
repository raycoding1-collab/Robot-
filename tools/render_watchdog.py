"""Keep an interleaved frame render alive until every frame is on disk and complete.

    python tools/render_watchdog.py --first 1 --last 501 --workers 2 --threads 2

Worker k renders frames first+k, first+k+workers, ... with `render.py --anim`.
That command skips frames that already exist, so a restarted worker resumes where
it stopped. Every minute the watchdog:

- restarts a worker whose process has died, after deleting any frame of its share
  left truncated by the crash;
- kills and restarts a worker that has produced no frame for --stall minutes;
- gives up (status FAILED) if one worker needs more than --max-restarts restarts,
  so a deterministic crash cannot loop forever;
- exits with status DONE once all frames exist and are complete PNGs.

Progress goes to renders/watchdog.log; renders/watchdog.status holds one word.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(ROOT, "renders", "frames")
LOG = os.path.join(ROOT, "renders", "watchdog.log")
STATUS = os.path.join(ROOT, "renders", "watchdog.status")
PNG_SIG = b"\x89PNG\r\n\x1a\n"


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S UTC ", time.gmtime()) + msg
    with open(LOG, "a") as f:
        f.write(line + "\n")


def set_status(word):
    with open(STATUS, "w") as f:
        f.write(word + "\n")


def frame_path(n):
    return os.path.join(FRAMES, f"f{n:04d}.png")


def png_complete(path):
    """True if the file starts with the PNG signature and ends with the IEND chunk."""
    try:
        if os.path.getsize(path) < 1024:
            return False
        with open(path, "rb") as f:
            head = f.read(8)
            f.seek(-12, os.SEEK_END)
            tail = f.read(12)
        return head == PNG_SIG and tail[4:8] == b"IEND"
    except OSError:
        return False


def worker_pids(start, step):
    """PIDs of `render.py --anim --start <start> --step <step>` processes running in ROOT."""
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        if len(argv) < 2 or not argv[1].endswith("render.py") or "--anim" not in argv:
            continue
        try:  # only workers of this checkout (render.py is run from ROOT)
            if os.path.realpath(f"/proc/{d}/cwd") != os.path.realpath(ROOT):
                continue
        except OSError:
            continue
        try:
            s = argv[argv.index("--start") + 1]
            st = argv[argv.index("--step") + 1]
        except (ValueError, IndexError):
            continue
        if s == str(start) and st == str(step):
            pids.append(int(d))
    return pids


def launch(python, start, step, threads, logname):
    out = open(os.path.join(ROOT, "renders", logname), "a")
    cmd = [python, "render.py", "--anim", "--start", str(start), "--step", str(step), "--threads", str(threads)]
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log(f"started worker start={start} pid={p.pid}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=501)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--stall", type=float, default=20.0, help="minutes without a new frame before a restart")
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--interval", type=float, default=60.0)
    a = ap.parse_args()

    os.makedirs(FRAMES, exist_ok=True)
    n_w = a.workers
    starts = [a.first + k for k in range(n_w)]
    share = {s: list(range(s, a.last + 1, n_w)) for s in starts}
    logname = {s: f"render_{chr(ord('a') + k)}.log" for k, s in enumerate(starts)}
    restarts = {s: 0 for s in starts}
    children = {}
    last_progress = {s: time.time() for s in starts}
    done_count = {s: -1 for s in starts}
    set_status("RUNNING")
    log(f"watchdog up: frames {a.first}-{a.last}, {n_w} workers")

    while True:
        all_done = True
        for s in starts:
            missing = [n for n in share[s] if not os.path.exists(frame_path(n))]
            count = len(share[s]) - len(missing)
            if count != done_count[s]:
                done_count[s] = count
                last_progress[s] = time.time()
            if s in children:
                children[s].poll()  # reap a finished child so it does not linger as a zombie
            pids = worker_pids(s, n_w)
            if pids:
                all_done = False
                if time.time() - last_progress[s] > a.stall * 60:
                    log(f"worker start={s} stalled for {a.stall:.0f} min; killing {pids}")
                    for pid in pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except OSError:
                            pass
                    time.sleep(3)
                    pids = []
                else:
                    continue
            # worker not running: drop frames it left truncated, then resume if work remains
            bad = [n for n in share[s] if os.path.exists(frame_path(n)) and not png_complete(frame_path(n))]
            for n in bad:
                os.remove(frame_path(n))
                log(f"removed incomplete frame {n}")
            missing = [n for n in share[s] if not os.path.exists(frame_path(n))]
            if not missing:
                continue
            all_done = False
            if restarts[s] >= a.max_restarts:
                log(f"worker start={s} exceeded {a.max_restarts} restarts; giving up with {len(missing)} frames missing")
                set_status("FAILED")
                return 1
            restarts[s] += 1
            log(f"worker start={s} not running with {len(missing)} frames left (next {missing[0]}); restart #{restarts[s]}")
            children[s] = launch(a.python, s, n_w, a.threads, logname[s])
            last_progress[s] = time.time()
        if all_done:
            bad = [n for n in range(a.first, a.last + 1) if not png_complete(frame_path(n))]
            if not bad:
                log(f"all {a.last - a.first + 1} frames complete")
                set_status("DONE")
                return 0
            for n in bad:
                if os.path.exists(frame_path(n)):
                    os.remove(frame_path(n))
            log(f"final check found {len(bad)} bad frames; re-rendering them")
            continue
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

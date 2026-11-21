"""Objective checks for renders/foley.wav against the simulation events.

    python tools/verify_sound.py [--dir renders]
Writes spectrogram/waveform PNGs next to the wav and prints a PASS/FAIL report.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import wave

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
import bpy  # noqa: F401,E402  (mathutils for robot_rubiks imports)
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from robot_rubiks import sound  # noqa: E402
from robot_rubiks.choreography import Choreography  # noqa: E402
from robot_rubiks.rig_def import FPS  # noqa: E402

SR = sound.SR


def read_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    assert sw == 3 and ch == 2 and sr == SR, (sw, ch, sr)
    b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
    v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
    v = np.where(v & 0x800000, v - 0x1000000, v)
    return (v.reshape(-1, 2).T / 8388607.0), sr


def onsets(x, band=(1000, 12000), hop_ms=1.0, win_ms=2.0, ratio_db=9.0, floor_db=-45.0, min_gap=0.010):
    """Robust transient detector: band energy jump vs. the median of the preceding 30 ms.

    Energy is measured in 2 ms windows: in 1 ms windows band-limited noise (motor whir, fan)
    swings by +-10 dB from one window to the next purely by chance, which is not a transient."""
    m = x.mean(axis=0)
    h = sound.bandpass(m, band[0], band[1], 4)
    hop = int(hop_ms / 1000 * SR)
    w = int(win_ms / 1000 * SR)
    p = np.convolve(h ** 2, np.ones(w) / w, mode="full")[w - 1:]  # trailing-edge aligned: window starts at the sample
    e = p[::hop][: int(np.ceil(len(h) / hop))]
    edb = 10 * np.log10(e + 1e-15)
    ref = np.array([np.median(edb[max(0, i - 30): max(1, i - 2)]) for i in range(len(edb))])
    top = edb.max()
    cand = (edb - ref > ratio_db) & (edb > top + floor_db)
    idx = np.flatnonzero(cand)
    # peak picking: within each group of candidates closer than min_gap keep the strongest
    # jump, so a small grain just before a click cannot mask the click itself
    jump = edb - ref
    groups = []
    for i in idx:
        if groups and (i - groups[-1][-1]) * hop_ms / 1000 <= min_gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    # time-stamp each event at the rising edge of its strongest transient: the earliest
    # candidate at most 3 ms before the strongest jump (an earlier grain is a separate event)
    out = []
    for g in groups:
        p = max(g, key=lambda i: jump[i])
        out.append(min(i for i in g if 0 <= (p - i) * hop_ms <= 3.0))
    return np.array(out) * hop / SR, edb


def whine(x, lo=150, hi=8000):
    """Largest steady tonal line in the film: (prominence dB over the median-filtered floor, time s)."""
    m = x.mean(axis=0)
    best = (0.0, 0.0)
    for t0 in np.arange(0, len(m) / SR - 3.0 + 1e-9, 1.5):
        seg = m[int(t0 * SR): int((t0 + 3.0) * SR)]
        fz, Pz = signal.welch(seg, SR, nperseg=16384)
        band = (fz > lo) & (fz < hi)
        pdb = 10 * np.log10(Pz[band] + 1e-20)
        p = float(np.max(pdb - signal.medfilt(pdb, 101)))
        if p > best[0]:
            best = (p, t0 + 1.5)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(__file__.rsplit("/tools/", 1)[0], "renders"))
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    path = os.path.join(a.dir, "foley.wav")
    x, sr = read_wav(path)
    c = Choreography()
    n_frames = int(c.duration * FPS)
    results = []

    def check(name, ok, detail):
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 1. format / duration
    dur = x.shape[1] / sr
    check("duration matches picture", abs(dur - n_frames / FPS) < 1 / sr + 1e-9, f"{dur:.5f}s vs {n_frames / FPS:.5f}s ({n_frames} frames)")
    check("finite samples", bool(np.all(np.isfinite(x))), "no NaN/Inf")
    dc = np.abs(x.mean(axis=1)).max()
    check("DC offset", dc < 1e-4, f"{dc:.2e}")
    peak = 20 * math.log10(np.abs(x).max())
    check("sample peak <= -1 dBFS", peak <= -1.0, f"{peak:.2f} dBFS")
    up = signal.resample_poly(x, 4, 1, axis=1)
    tp = 20 * math.log10(np.abs(up).max())
    check("true peak <= -1 dBTP", tp <= -1.0, f"{tp:.2f} dBTP (4x oversampled)")
    clip = int(np.sum(np.abs(x) > 0.999))
    check("no clipped samples", clip == 0, f"{clip}")
    lufs = sound.integrated_lufs(x)
    check("integrated loudness (EBU R128 short-form)", abs(lufs + 23) < 0.5, f"{lufs:.2f} LUFS (target -23)")
    # independent loudness measurement via ffmpeg's EBU R128 meter
    try:
        r = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
        tail = r.stderr[r.stderr.rfind("Summary:"):]
        I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", tail).group(1))
        TP = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail).group(1))
        check("ffmpeg EBU R128 agrees", abs(I - lufs) < 1.0 and TP <= -1.0, f"I={I} LUFS, TP={TP} dBFS")
    except Exception as e:  # pragma: no cover
        check("ffmpeg EBU R128 agrees", False, str(e))
    # 2. discontinuities (pops): ultrasonic-band spikes away from any scheduled impact
    ev_all = np.array([e["t"] for e in json.load(open(os.path.join(a.dir, "foley_events.json")))])
    hf = sound.highpass(x.mean(axis=0), 17000, 6)
    env_hf = np.abs(hf)
    win = int(0.002 * SR)
    loc = np.sqrt(np.convolve(hf ** 2, np.ones(win * 10) / (win * 10), mode="same")) + 1e-9
    spikes = np.flatnonzero((env_hf > 12 * loc) & (env_hf > 1e-3))  # audible (> -60 dBFS) ultrasonic spikes
    spikes = [i for i in spikes if np.min(np.abs(ev_all - i / SR)) > 0.010]
    if spikes:
        print("      spike times:", [round(i / SR, 4) for i in spikes[:12]])
    check("no discontinuities / pops", len(spikes) == 0, f"{len(spikes)} ultrasonic spikes outside impact events")
    # 3. stereo sanity
    corr = np.corrcoef(x[0], x[1])[0, 1]
    check("mono-compatible stereo", corr > 0.2, f"L/R correlation {corr:.2f}")
    # silence before power-on should be near room tone only
    pre = x[:, : int(0.2 * sr)]
    pre_db = 20 * math.log10(np.sqrt(np.mean(pre ** 2)) + 1e-12)
    check("room tone audible but quiet", -75 < pre_db < -45, f"{pre_db:.1f} dBFS RMS before power-on")

    # 4. sync: every simulated magnet capture must produce a click within 4 ms
    ev = json.load(open(os.path.join(a.dir, "foley_events.json")))
    clacks = np.array([e["t"] for e in ev if e["kind"] == "clack"])
    det, env = onsets(x)
    offs = []
    missed = []
    for t in clacks:
        if len(det) == 0:
            missed.append(t)
            continue
        k = np.argmin(np.abs(det - t))
        o = det[k] - t
        if abs(o) <= 0.004:
            offs.append(o)
        else:
            missed.append(t)
    offs = np.array(offs) * 1000
    check("every layer-turn click detected", len(missed) == 0, f"{len(clacks) - len(missed)}/{len(clacks)} found" + (f", missed at {[round(m, 3) for m in missed]}" if missed else ""))
    if len(offs):
        check("click timing vs physics", np.max(np.abs(offs)) <= 3.0, f"mean {offs.mean():+.2f} ms, max |{np.max(np.abs(offs)):.2f}| ms")
    # turns vs picture: capture instants fall on the frames where the layer reaches 90 deg
    s = c.solve
    bad = 0
    for tr in s.turns:
        tc = [e for e in clacks if tr.t0 <= e <= tr.t_end + 0.01]
        if not tc:
            bad += 1
            continue
        ang = abs(tr.angle(tc[0]))
        if not (math.radians(85) <= ang <= math.radians(95)):
            bad += 1
    check("click occurs when the layer reaches 90 deg in the animation", bad == 0, f"{len(s.turns) - bad}/{len(s.turns)} turns")
    # unexpected transients: detected onsets not explained by any scheduled event
    # physically active intervals: impacts ring for ~25 ms, a sliding layer sounds until its magnet capture
    active = []
    for e in ev:
        if e["kind"] == "slide":
            nxt = clacks[clacks > e["t"]]
            active.append((e["t"] - 0.003, (nxt[0] if len(nxt) else e["t"] + 0.6) + 0.025))
        else:
            active.append((e["t"] - 0.005, e["t"] + 0.025))
    stray = [t for t in det if not any(a0 <= t <= a1 for a0, a1 in active)]
    check("no unexplained transients", len(stray) == 0, f"{len(stray)} stray onsets" + (f" at {[round(t, 3) for t in stray[:10]]}" if stray else ""))

    # 5. mix balance from the stems (all at master gain)
    stems = dict(np.load(os.path.join(a.dir, "foley_stems.npz")))
    def pk(v):
        return 20 * math.log10(np.abs(v).max() + 1e-12)
    def rms_win(v, t0, t1):
        i0, i1 = int(t0 * SR), int(t1 * SR)
        return 20 * math.log10(np.sqrt(np.mean(v[:, i0:i1] ** 2)) + 1e-12)
    click_peaks = []
    for t in clacks:
        i = int(t * SR)
        click_peaks.append(20 * math.log10(np.abs(stems["cube_click"][:, i: i + int(0.02 * SR)]).max() + 1e-12))
    cp = float(np.median(click_peaks))
    sl = stems["cube_slide"]
    slide_db = max(20 * math.log10(np.sqrt(np.mean(sl[:, int(tr.t0 * SR): int((tr.t0 + 0.04) * SR)] ** 2)) + 1e-12) for tr in s.turns)
    serv = rms_win(stems["servo"], s.t_start, s.t_end)
    serv_pk_win = max(rms_win(stems["servo"], t, t + 0.1) for t in np.arange(1.0, c.duration - 0.2, 0.05))
    tap_pk = pk(stems["finger_tap"])
    room = rms_win(stems["room_tone"], 0.5, c.duration - 0.5)
    print(f"      click peak (median) {cp:.1f} dBFS | slide rms {slide_db:.1f} | servo rms (solve) {serv:.1f}, loudest 100 ms {serv_pk_win:.1f} | tap peak {tap_pk:.1f} | room tone rms {room:.1f}")
    check("clicks are the loudest cube sound", -14 <= cp <= -2, f"median click peak {cp:.1f} dBFS")
    check("slide noise sits under the clicks", -32 <= slide_db - cp <= -14, f"{slide_db - cp:+.1f} dB re click peak")
    check("servo whir is subtle", serv_pk_win - cp <= -20, f"loudest servo 100 ms {serv_pk_win - cp:+.1f} dB re click peak")
    check("finger taps softer than clicks", -24 <= tap_pk - cp <= -3, f"{tap_pk - cp:+.1f} dB re click peak")
    check("room tone level", -66 <= room <= -50, f"{room:.1f} dBFS RMS")

    # tonal whine check over the WHOLE film (3 s windows, 1.5 s apart): no steady narrow-band line
    # more than 12 dB above the local spectral floor between 150 Hz and 8 kHz
    prom, at = whine(x)
    check("no audible steady whine anywhere (150 Hz - 8 kHz)", prom < 12, f"max tonal prominence {prom:.1f} dB at {at:.1f} s (limit 12)")

    # 6. listening comfort and realism of the cube sound
    def kdb(v, t0, t1):
        y = sound.k_weight(v[:, int(t0 * SR): int(t1 * SR)])
        return -0.691 + 10 * math.log10(np.sum(np.mean(y ** 2, axis=1)) + 1e-20)
    insp0, solve0, solve1 = 3.0, s.t_start - 0.3, s.t_end + 0.2
    jump = kdb(x, solve0, solve1) - kdb(x, insp0, solve0)
    # The inspection is a robot quietly turning a cube in rubber fingers in a quiet room, so it is
    # near-silent; lifting it to narrow this contrast is what made the handling and motor sounds
    # painful. This only guards against extremes (a quiet room vs. a cube at 0.5 m is ~20 dB).
    check("no extreme loudness jump into the solve", jump <= 22.0, f"solve is {jump:+.1f} LU above the inspection (limit +22)")
    # Outside the solve nothing may come close to the level the listener sets the volume by, and
    # the robot's own motors must stay a faint texture. Momentary (400 ms) loudness, re integrated.
    # (The version that sounded painful throughout had the cube set-down at +0.4 LU and motor
    # whine all through the film.)
    def momentary_max(v, outside=(solve0, solve1)):
        yk = sound.k_weight(v)
        best = (-200.0, 0.0)
        for t0 in np.arange(0, v.shape[1] / SR - 0.4, 0.05):
            if outside[0] - 0.4 < t0 < outside[1]:
                continue
            l = -0.691 + 10 * math.log10(np.sum(np.mean(yk[:, int(t0 * SR): int((t0 + 0.4) * SR)] ** 2, axis=1)) + 1e-20)
            best = max(best, (l, t0 + 0.2))
        return best[0] - lufs, best[1]
    l, t = momentary_max(x)
    check("nothing loud outside the solve", l <= -10.0, f"loudest 400 ms outside the solve {l:+.1f} LU re programme at {t:.2f} s (limit -10)")
    l, t = momentary_max(stems["servo"], outside=(-1, -1))
    check("robot motors are a faint texture", l <= -16.0, f"loudest 400 ms of motor sound {l:+.1f} LU re programme at {t:.2f} s (limit -16)")
    # each click must be a short damped plastic contact, not a ringing tone
    rings, qs = [], []
    ck = stems["cube_click"].mean(axis=0)
    for t in clacks:
        i = int(t * SR)
        y = ck[i: i + int(0.08 * SR)]
        env = np.sqrt(np.convolve(y ** 2, np.ones(48) / 48, mode="same"))
        p = int(np.argmax(env))
        below = np.flatnonzero(env[p:] < env[p] * 0.01)
        rings.append(1000 * below[0] / SR if len(below) else 80.0)
        # quality factor of the strongest resonance: Q = f0 / (-3 dB bandwidth).
        # A hand-held plastic cubie is damped to Q ~ 10-20; a small bell rings at Q ~ 100.
        fz, Pz = signal.periodogram(np.pad(y, (0, 32768 - len(y))), SR)
        bz = np.flatnonzero((fz > 500) & (fz < 8000))
        k0 = bz[np.argmax(Pz[bz])]
        half = Pz[k0] / 2
        lo, hi = k0, k0
        while lo > 0 and Pz[lo] > half:
            lo -= 1
        while hi < len(Pz) - 1 and Pz[hi] > half:
            hi += 1
        qs.append(fz[k0] / max(fz[hi] - fz[lo], 1e-9))
    check("clicks are short contacts, not ringing tones", np.median(rings) <= 25 and np.median(qs) <= 30,
          f"ring-down to -40 dB {np.median(rings):.1f} ms (limit 25), strongest resonance Q {np.median(qs):.0f} (limit 30)")
    # the turning sound is stick-slip friction (impulsive grains), not smooth hiss. Kurtosis is
    # measured after dividing out the 5 ms RMS envelope, so the speed envelope does not count:
    # Gaussian hiss gives ~3 whatever its envelope.
    kur = []
    w5 = int(0.005 * SR)
    for tr in s.turns:
        v = stems["cube_slide"].mean(axis=0)[int(tr.t0 * SR): int((tr.t0 + 0.05) * SR)]
        if np.std(v) > 0:
            u = v / (np.sqrt(np.convolve(v ** 2, np.ones(w5) / w5, mode="same")) + 1e-12)
            u = u[w5:-w5]
            kur.append(np.mean((u - u.mean()) ** 4) / np.var(u) ** 2)
    check("turning sound is grainy plastic friction, not hiss", np.median(kur) >= 4.5, f"median envelope-normalised kurtosis {np.median(kur):.1f} (smooth hiss ~3, limit 4.5)")
    # no single pitch region dominates the solve (the old ringing clicks put ~40 % of the energy near 1.9 kHz)
    seg = x.mean(axis=0)[int(solve0 * SR): int(solve1 * SR)]
    fz, Pz = signal.welch(seg, SR, nperseg=8192)
    centers = 1000 * 2 ** (np.arange(-12, 13) / 3)
    shares = [Pz[(fz >= fc * 2 ** (-1 / 6)) & (fz < fc * 2 ** (1 / 6))].sum() / Pz[fz >= 20].sum() for fc in centers]
    k = int(np.argmax(shares))
    check("no dominant pitch region in the solve", shares[k] <= 0.20, f"largest third-octave band {100 * shares[k]:.1f} % of energy at {centers[k]:.0f} Hz (limit 20 %)")

    # 7. pictures
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m = x.mean(axis=0)
        fig, axs = plt.subplots(3, 1, figsize=(16, 11))
        f, t, S = signal.spectrogram(m, SR, nperseg=2048, noverlap=1536)
        axs[0].pcolormesh(t, f, 10 * np.log10(S + 1e-14), shading="auto", vmin=-150, vmax=-50, cmap="magma")
        axs[0].set_yscale("symlog", linthresh=200)
        axs[0].set_ylim(30, 24000)
        axs[0].set_title("Full mix spectrogram")
        for k, v in c.T.items():
            if isinstance(v, float):
                axs[0].axvline(v, color="cyan", lw=0.4)
        t0, t1 = s.t_start - 0.1, s.t_start + 1.6
        i0, i1 = int(t0 * SR), int(t1 * SR)
        f2, t2, S2 = signal.spectrogram(m[i0:i1], SR, nperseg=512, noverlap=448)
        axs[1].pcolormesh(t2 + t0, f2, 10 * np.log10(S2 + 1e-14), shading="auto", vmin=-150, vmax=-50, cmap="magma")
        for tt in clacks:
            if t0 < tt < t1:
                axs[1].axvline(tt, color="cyan", lw=0.6)
        axs[1].set_title("Solve (first 1.6 s): cyan = simulated magnet captures")
        tt = np.arange(i0, i1) / SR
        axs[2].plot(tt, x[0, i0:i1], lw=0.4, label="L")
        axs[2].plot(tt, x[1, i0:i1], lw=0.4, alpha=0.7, label="R")
        for e in clacks:
            if t0 < e < t1:
                axs[2].axvline(e, color="k", lw=0.4)
        axs[2].legend()
        axs[2].set_title("Waveform (solve)")
        plt.tight_layout()
        plt.savefig(os.path.join(a.dir, "foley_analysis.png"), dpi=80)
        # single event zoom
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        e = clacks[len(clacks) // 2]
        j0, j1 = int((e - 0.03) * SR), int((e + 0.08) * SR)
        axs[0].plot(np.arange(j0, j1) / SR - e, m[j0:j1], lw=0.6)
        axs[0].set_title("One layer-turn click (t=0: magnet capture)")
        fz, Pz = signal.welch(m[int(e * SR): int((e + 0.05) * SR)], SR, nperseg=1024)
        axs[1].semilogx(fz, 10 * np.log10(Pz + 1e-16))
        axs[1].set_xlim(100, 20000)
        axs[1].set_title("Click spectrum")
        plt.tight_layout()
        plt.savefig(os.path.join(a.dir, "foley_click.png"), dpi=80)
    except Exception as ex:
        print("plotting skipped:", ex)
    n_fail = sum(1 for r in results if not r[1])
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return n_fail


if __name__ == "__main__":
    sys.exit(main())

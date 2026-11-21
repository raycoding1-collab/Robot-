"""Physically driven foley, generated from the same choreography the picture uses.

Every sound is placed by the simulation itself (sample-accurate):
  * cube layer turns: plastic slide noise scaled by the layer's angular speed and
    a modal-synthesis "clack" at the instant the magnets capture the layer;
  * fingertip taps (rubber pad on plastic) at each contact / regrip;
  * cube-on-desk contact, hands settling on the desk, lift-off;
  * servo/gear whine for every joint group, pitch and level driven by the
    actual joint angular velocities of the rig;
  * cooling fan, power-on relay, room tone (HVAC + distant city through glass);
  * microphone perspective follows the edit (distance law + stereo position
    relative to the active camera), plus a convolution room reverb.
"""
from __future__ import annotations

import math
import wave

import numpy as np
from scipy import signal

from . import physics as P
from .rig_def import FPS

SR = 48000


# ----------------------------------------------------------------------------- dsp helpers
def _sos(kind, f, order=4):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")


def bandpass(x, lo, hi, order=4):
    return signal.sosfilt(_sos("bandpass", [lo, hi], order), x)


def highpass(x, f, order=4):
    return signal.sosfilt(_sos("highpass", f, order), x)


def lowpass(x, f, order=4):
    return signal.sosfilt(_sos("lowpass", f, order), x)


def addp(a, b, gb=1.0):
    """a + gb*b, zero-padding the shorter signal."""
    n = max(len(a), len(b))
    out = np.zeros(n)
    out[: len(a)] += a
    out[: len(b)] += gb * b
    return out


def db(x):
    return 10 ** (x / 20)


def modal(rng, freqs, taus, amps, dur=0.12, jitter=0.035, noise=0.35, noise_ms=0.9, noise_band=(1800, 14000), attack_ms=0.08):
    """Impact synthesis: damped sinusoidal modes + a short contact-noise burst."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, tau, a in zip(freqs, taus, amps):
        ff = f * (1 + rng.normal(0, jitter))
        aa = a * (1 + rng.normal(0, 0.15))
        tt = tau * (1 + rng.normal(0, 0.1))
        y += aa * np.exp(-t / tt) * np.sin(2 * np.pi * ff * t)
    if noise > 0:
        nb = rng.normal(0, 1, n) * np.exp(-t / (noise_ms / 1000))
        nb = bandpass(nb, noise_band[0], min(noise_band[1], SR / 2 - 100), 2)
        nb /= np.max(np.abs(nb)) + 1e-12
        y += noise * nb
    y *= 1 - np.exp(-t / (attack_ms / 1000))
    y /= np.max(np.abs(y)) + 1e-12
    return y


def impact(rng, freqs, taus, amps, contact_ms=0.4, dur=0.06, jitter=0.06, grit=0.12, grit_band=(700, 5500)):
    """Hertzian impact: a half-sine contact force of finite duration excites damped modes.

    The contact time low-passes the excitation (spectrum falls off above ~1/contact),
    so a soft, short-lived plastic contact gives a "tock" instead of a bright ping.
    """
    n = int(dur * SR)
    t = np.arange(n) / SR
    modes = np.zeros(n)
    for f, tau, a in zip(freqs, taus, amps):
        ff = f * (1 + rng.normal(0, jitter))
        tt = tau * (1 + rng.normal(0, 0.12))
        aa = a * (1 + rng.normal(0, 0.2))
        modes += aa * np.exp(-t / tt) * np.sin(2 * np.pi * ff * t + rng.uniform(-0.3, 0.3))
    k = max(2, int(contact_ms / 1000 * SR))
    force = np.sin(np.pi * np.arange(k) / k)
    y = signal.fftconvolve(modes, force)[:n]
    if grit > 0:  # surface roughness of the sliding faces at the moment of capture
        g = rng.normal(0, 1, n) * np.exp(-t / 0.0015)
        g = bandpass(g, grit_band[0], grit_band[1], 2)
        y = y / (np.max(np.abs(y)) + 1e-12) + grit * g / (np.max(np.abs(g)) + 1e-12)
    fade = np.ones(n)
    m = int(0.006 * SR)
    fade[-m:] = np.linspace(1, 0, m)
    y *= fade
    return y / (np.max(np.abs(y)) + 1e-12)


# A cubie is a ~19 mm hollow ABS shell with ~1.3 mm walls. Its lowest shell modes lie between
# about 1.3 and 6 kHz. Pressed against its neighbours and held in two hands, it is damped to
# Q ~ 12-18, so the modes die within a few milliseconds.
CUBIE_MODES = dict(freqs=[1350, 2250, 3250, 4550, 6100], taus=[0.0032, 0.0023, 0.0016, 0.0011, 0.0008], amps=[0.7, 1.0, 0.85, 0.55, 0.3])
# The whole cube in the hands: hollow 57 mm body plus core, heavily damped (the "thock").
CUBE_BODY = dict(freqs=[540, 900, 1450], taus=[0.0045, 0.0032, 0.0022], amps=[0.8, 0.6, 0.35])


def cubie_contact(rng, contact_ms, dur=0.03, grit=0.12):
    """One hard plastic-on-plastic contact exciting one cubie (each cubie is slightly different)."""
    scale = rng.uniform(0.85, 1.15)
    m = dict(freqs=[f * scale for f in CUBIE_MODES["freqs"]], taus=CUBIE_MODES["taus"], amps=CUBIE_MODES["amps"])
    return impact(rng, **m, contact_ms=contact_ms, dur=dur, jitter=0.05, grit=grit, grit_band=(1500, 9000))


def capture_click(rng, speed=0.5, held=True):
    """Magnets pull a turning layer into alignment: 2-4 of its cubies seat against their
    neighbours within ~3 ms, so one capture is a short cluster of hard contacts plus the
    damped body of the cube. speed 0..1 = how fast the layer arrives."""
    n = int(0.05 * SR)
    y = np.zeros(n)
    k = int(rng.integers(2, 5))
    offs = np.sort(rng.uniform(0, 0.0028, k))
    offs[0] = 0.0
    for i, o in enumerate(offs):
        a = 1.0 if i == 0 else rng.uniform(0.25, 0.7)
        # Hertz contact time of a ~10 g ABS cubie meeting its neighbour: ~0.43 ms at 0.3 m/s,
        # ~0.35 ms at the 0.85 m/s edge speed of a fast flick (t ~ v^-1/5)
        c = cubie_contact(rng, contact_ms=rng.uniform(0.26, 0.38) - 0.06 * speed)
        j = int(o * SR)
        m = min(len(c), n - j)
        y[j: j + m] += a * c[:m]
    if held:
        y = addp(y, impact(rng, **CUBE_BODY, contact_ms=0.6, dur=0.04, grit=0.0), 0.45)
    y = lowpass(y, 9000, 2)  # air absorption + microphone
    return y / (np.max(np.abs(y)) + 1e-12)


def scrape(rng, w):
    """Sliding layer: stick-slip friction of plastic on plastic. A Poisson train of tiny contacts
    whose rate and strength follow the layer's angular speed w (rad/s, per sample)."""
    n = len(w)
    drive = np.clip(w / 30.0, 0, 1.4)
    rate = 250 + 1350 * np.minimum(drive, 1.0)  # distinct slips per second
    idx = np.flatnonzero(rng.random(n) < rate / SR)
    amp = rng.gamma(2.0, 0.5, len(idx)) * drive[idx] ** 1.1
    out = np.zeros(n)
    grp = rng.integers(0, 4, len(idx))
    for g in range(4):  # a few different contact patches
        kern = cubie_contact(rng, contact_ms=0.09, dur=0.006, grit=0.0)
        a = np.zeros(n)
        a[idx[grp == g]] = amp[grp == g]
        out += signal.fftconvolve(a, kern)[:n]
    rub = bandpass(rng.normal(0, 1, n + 2048), 900, 6000, 2)[2048:]
    rub /= np.std(rub) + 1e-9
    out += 0.18 * rub * drive
    return lowpass(out, 10000, 2)


def make_ir(rng, rt60=0.55, dur=1.1, predelay_ms=4.0):
    """Stereo room impulse response: early reflections + frequency-dependent decaying tail."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    irs = []
    for ch in range(2):
        tail = rng.normal(0, 1, n) * np.exp(-6.91 * t / rt60)
        bright = tail
        dark = lowpass(tail, 2500, 2)
        mix = np.exp(-t / 0.12)
        tail = bright * mix + dark * (1 - mix)
        tail[: int(predelay_ms / 1000 * SR)] = 0
        er = np.zeros(n)
        for ms, g in ((2.6, 0.55), (7.9, 0.35), (12.4, 0.30), (17.8 + ch * 1.3, 0.22), (26.0 - ch * 2.1, 0.18), (39.0, 0.12)):
            k = int(ms / 1000 * SR)
            er[k] += g * (1 if rng.random() > 0.3 else -1)
        ir = 0.35 * tail / np.sqrt(np.sum(tail ** 2)) + er
        irs.append(ir)
    return np.array(irs)


class Mixer:
    def __init__(self, duration):
        self.n = int(round(duration * SR))
        self.dry = np.zeros((2, self.n))
        self.send = np.zeros((2, self.n))  # to reverb
        self.amb = np.zeros((2, self.n))  # room tone (no extra reverb)
        self.stems = {}
        self.stem = "misc"

    def _stem(self):
        if self.stem not in self.stems:
            self.stems[self.stem] = np.zeros((2, self.n))
        return self.stems[self.stem]

    def place(self, sig, t, gain, pan, wet=0.18):
        """Equal-power pan (-1 left .. +1 right)."""
        i0 = int(round(t * SR))
        if i0 >= self.n or gain <= 0:
            return
        sig = sig[: self.n - i0] if i0 >= 0 else sig[-i0:]
        i0 = max(0, i0)
        a = (pan + 1) * math.pi / 4
        gl, gr = math.cos(a) * gain, math.sin(a) * gain
        self.dry[0, i0 : i0 + len(sig)] += gl * sig
        self.dry[1, i0 : i0 + len(sig)] += gr * sig
        st = self._stem()
        st[0, i0 : i0 + len(sig)] += gl * sig
        st[1, i0 : i0 + len(sig)] += gr * sig
        self.send[0, i0 : i0 + len(sig)] += gl * wet * sig
        self.send[1, i0 : i0 + len(sig)] += gr * wet * sig

    def place_stream(self, sig, gain_curve, pan_curve, wet=0.2):
        """Continuous sound with per-sample gain/pan curves (same length as the mix)."""
        a = (np.clip(pan_curve, -1, 1) + 1) * np.pi / 4
        l = sig * np.cos(a) * gain_curve
        r = sig * np.sin(a) * gain_curve
        self.dry[0] += l
        self.dry[1] += r
        st = self._stem()
        st[0] += l
        st[1] += r
        self.send[0] += wet * l
        self.send[1] += wet * r


# ----------------------------------------------------------------------------- perspective
class Perspective:
    """Microphone = active camera. Gain by distance, pan by image position."""

    def __init__(self, shots):
        from .cameras import camera_at

        self.shots = shots
        self.cam_at = camera_at

    def gain_pan(self, p, t, d_ref=0.75):
        from mathutils import Vector

        cpos, ctgt, lens = self.cam_at(self.shots, t)
        v = Vector(p) - cpos
        d = max(0.15, v.length)
        fwd = (ctgt - cpos).normalized()
        right = fwd.cross(Vector((0, 0, 1))).normalized()
        x = v.dot(right) / max(0.05, v.dot(fwd))
        pan = max(-1.0, min(1.0, x / 0.55)) * 0.65
        # film-style perspective: closer is louder, but much gentler than the free-field
        # 1/d law, so cuts between close-ups and wides do not jump in level
        g = (d_ref / d) ** 0.5
        return max(0.5, min(1.4, g)), pan


# ----------------------------------------------------------------------------- the score
class Foley:
    SCRAPE_GAIN = 0.034
    PAD_TAP = dict(freqs=[820, 1450, 2380], taus=[0.0045, 0.0032, 0.0022], amps=[1.0, 0.6, 0.3])
    DESK = dict(freqs=[150, 265, 410, 630, 980], taus=[0.045, 0.032, 0.022, 0.015, 0.010], amps=[0.9, 0.7, 0.5, 0.35, 0.2])

    def __init__(self, choreo, shots, seed=11):
        self.c = choreo
        self.rng = np.random.default_rng(seed)
        self.dur = int(choreo.duration * FPS) / FPS
        self.mix = Mixer(self.dur)
        self.persp = Perspective(shots)
        self.events = []  # (time, kind, detail) — for verification

    # ---------------------------------------------------------------- primitives
    def _cube_pos(self, t):
        return self.c.cube_world(t).translation

    def clack(self, t, strength, pos=None, held=True):
        speed = min(1.0, max(0.0, (strength - 0.55) / 0.5))  # 0 = gentle capture .. 1 = fast capture
        y = capture_click(self.rng, speed, held)
        g, pan = self.persp.gain_pan(pos if pos is not None else self._cube_pos(t), t)
        self.mix.stem = "cube_click"
        self.mix.place(y, t, 0.30 * strength * g, pan, wet=0.16)
        self.events.append((t, "clack", strength))

    def tap(self, t, strength, pos=None, kind="pad"):
        r = self.rng
        y = modal(r, **self.PAD_TAP, dur=0.04, noise=0.25, noise_ms=0.5, noise_band=(600, 6000), attack_ms=0.25)
        y = lowpass(y, 5200, 2)
        g, pan = self.persp.gain_pan(pos if pos is not None else self._cube_pos(t), t)
        self.mix.stem = "finger_tap"
        self.mix.place(y, t, 0.10 * strength * g, pan, wet=0.12)
        self.events.append((t, "tap", strength))

    def desk_hit(self, t, strength, pos, with_cube=False, hand=False):
        r = self.rng
        y = modal(r, **self.DESK, dur=0.35, noise=0.4, noise_ms=1.6, noise_band=(300, 6000), attack_ms=0.3)
        if with_cube:
            y = addp(0.55 * y, capture_click(r, 0.3, held=False), 0.45)
        if hand:  # robot fingertip pads + shell on the desk
            y = addp(0.7 * lowpass(y, 2500, 2), modal(r, **self.PAD_TAP, dur=0.04, noise=0.2), 0.25)
        g, pan = self.persp.gain_pan(pos, t)
        self.mix.stem = "desk"
        self.mix.place(y, t, 0.30 * strength * g, pan, wet=0.22)
        self.events.append((t, "desk", strength))

    def slide(self, t0, t1, omega_fn, pos, strength=1.0, grip=False):
        """Sound of something sliding, level and texture following the angular speed.

        Layers: plastic stick-slip scrape (granular). Grip (rubber pads on the cube while
        the whole cube turns in the hands): a faint, dull rustle."""
        n = int((t1 - t0) * SR)
        if n <= 8:
            return
        ts = t0 + np.arange(n) / SR
        w = np.array([abs(omega_fn(x)) for x in ts[::48]])
        w = np.interp(np.arange(n), np.arange(0, n, 48)[: len(w)], w)
        g, pan = self.persp.gain_pan(pos, t0)
        if grip:
            env = np.clip(w / 30.0, 0, 1.4) ** 1.25
            nz = bandpass(self.rng.normal(0, 1, n + 2048), 500, 3500, 2)[2048:]
            nz /= np.std(nz) + 1e-9
            self.mix.stem = "cube_slide"
            self.mix.place(0.016 * strength * env * nz, t0, g, pan, wet=0.2)
        else:
            self.mix.stem = "cube_slide"
            self.mix.place(self.SCRAPE_GAIN * strength * scrape(self.rng, w), t0, g, pan, wet=0.2)
        self.events.append((t0, "slide", strength))

    # ---------------------------------------------------------------- score sections
    def cube_turns(self):
        s = self.c.solve
        for tr in s.turns:
            prof = tr.prof
            om = lambda x, tr=tr: (tr.prof(x - tr.t0 + 5e-4) - tr.prof(x - tr.t0 - 5e-4)) / 1e-3
            # magnet capture: first time the layer reaches ~89.3 deg
            tc = None
            for k in range(1, 4000):
                x = k * 5e-5
                if prof(x) >= math.radians(89.3):
                    tc = x
                    break
            tc = tc if tc is not None else prof.duration
            w_c = abs(om(tr.t0 + tc))
            strength = 0.55 + 0.5 * min(1.0, w_c / 25.0)
            self.slide(tr.t0, tr.t0 + tc + 0.004, om, self._cube_pos(tr.t0), strength=0.9 + 0.3 * min(1, w_c / 25))
            self.clack(tr.t0 + tc, strength)
            # tiny second tick when a flicked layer settles back from its overshoot
            if tr.kind == "flick":
                peak_t = max(range(len(prof.th)), key=lambda i: prof.th[i])
                self.tap(tr.t0 + prof.ts[peak_t] + 0.012, 0.25)
                self.tap(tr.t0 - 0.002, 0.55)  # fingertip strikes the cubie
            elif tr.kind == "twist":
                peak_t = max(range(len(prof.th)), key=lambda i: prof.th[i])
                self.tap(tr.t0 + prof.ts[peak_t] + 0.012, 0.22)
                self.tap(tr.t_release + tr.regrip - 0.004, 0.35)  # pads re-seat
            elif tr.kind == "wrist":
                self.tap(tr.t_free - 0.004, 0.35)  # pads re-seat after the regrip
                self.tap(tr.t_free + 0.006, 0.25)

    def handling(self):
        T = self.c.T
        hand_pos = lambda side, t: self.c.hand(side, t)[0].translation
        # hands lift off the desk at the start of the reach
        for side, dt in (("R", 0.0), ("L", 0.035)):
            self.desk_hit(T["reach0"] + 0.02 + dt, 0.10, hand_pos(side, T["reach0"]), hand=True)
        # grasp: thumbs and middles land on the cube
        for k, dt in enumerate((-0.045, -0.030, -0.018, -0.006)):
            self.tap(T["grasp"] + dt, 0.75 - 0.1 * k)
        # cube leaves the desk
        self.desk_hit(T["lift0"] + 0.03, 0.06, self._cube_pos(T["lift0"]), with_cube=True)
        # inspection regrips: pads release and re-seat
        for ins in T["insp"]:
            for key in ("regripR", "regripL"):
                g0, g1 = ins[key]
                self.tap(g0 + 0.015, 0.22)
                self.tap(g1 - 0.010, 0.5)
            r0, r1 = ins["rot"]
            # the whole cube turning in the hands: faint grip creak
            self.slide(r0, r1, lambda x, r0=r0, r1=r1: P.QUARTER * P.min_jerk_d((x - r0) / (r1 - r0)) / (r1 - r0) * 3.0, self._cube_pos(r0), strength=1.6, grip=True)
        # set down on the desk
        t_down = T["down"][1]
        self.desk_hit(t_down - 0.004, 0.55, self._cube_pos(t_down), with_cube=True)
        for k, dt in enumerate((0.03, 0.05, 0.07, 0.09)):
            self.tap(T["release"][0] + dt, 0.18)
        # hands come to rest on the desk
        for side, dt in (("R", 0.0), ("L", 0.055)):
            self.desk_hit(T["retract"][1] - 0.01 + dt, 0.22, hand_pos(side, T["retract"][1]), hand=True)

    def servos(self, rig):
        """Gear/motor whine from the rig's joint angular velocities."""
        from mathutils import Matrix

        rate = 240
        n_ctl = int(self.dur * rate) + 2
        ts = np.arange(n_ctl) / rate
        groups = {
            "arm_R": (["uparm_R", "forearm_R", "wroll_R", "hand_R"], 210.0, 0, db(-38)),
            "arm_L": (["uparm_L", "forearm_L", "wroll_L", "hand_L"], 196.0, 0, db(-38)),
            "fing_R": ([f"{f}{k}_R" for f in ("thumb", "index", "middle", "ring", "pinky") for k in (1, 2, 3)], 560.0, 0, db(-52)),
            "fing_L": ([f"{f}{k}_L" for f in ("thumb", "index", "middle", "ring", "pinky") for k in (1, 2, 3)], 530.0, 0, db(-52)),
            "head": (["neck", "head"], 290.0, 0, db(-40)),
            "torso": (["spine", "chest"], 140.0, 0, db(-50)),
        }
        prev = None
        speeds = {g: np.zeros(n_ctl) for g in groups}
        pos = {g: np.zeros((n_ctl, 3)) for g in groups}
        for i, t in enumerate(ts):
            pose = rig.evaluate(min(t, self.c.duration - 1e-3))
            if prev is not None:
                for g, (names, *_r) in groups.items():
                    s = 0.0
                    for nme in names:
                        q0 = prev[nme].to_quaternion()
                        q1 = pose[nme].to_quaternion()
                        ang = q0.rotation_difference(q1).angle
                        s += min(ang * rate, 60.0)
                    speeds[g][i] = s / math.sqrt(len(names))
            prev = pose
            for g in groups:
                pos[g][i] = (0, -0.25, 1.0)
        n = self.mix.n
        tt = np.arange(n) / SR
        self.mix.stem = "servo"
        for g, (names, f0, k, gain) in groups.items():
            sp = np.convolve(speeds[g], np.ones(7) / 7, mode="same")
            s = np.interp(tt, ts, sp)
            # BLDC + strain-wave gear: pitch rises with speed but saturates (bounded range)
            f = f0 * (1.0 + 1.6 * (1.0 - np.exp(-s / 5.0)))
            ph = 2 * np.pi * np.cumsum(f) / SR
            wob = 0.003 * np.sin(2 * np.pi * 6.1 * tt + self.rng.uniform(0, 6))
            tone = np.sin(ph * (1 + wob)) + 0.35 * np.sin(2 * ph + 0.4) + 0.14 * np.sin(3 * ph + 1.1) + 0.07 * np.sin(4.73 * ph)
            whir = bandpass(self.rng.normal(0, 1, n), 300, 2600, 2)
            whir /= np.std(whir)
            # gear mesh: the tooth impacts modulate the mechanical noise at the mesh rate, which
            # gives a buzzy whir rather than a pure electronic tone
            whir *= 1.0 + 0.35 * np.sin(ph)
            amp = (s / (s + 4.0)) ** 1.2
            # rotor + gear-train inertia: the whir builds up over ~20 ms and decays over ~40 ms
            amp = _attack_release(amp, attack_s=0.020, release_s=0.040)
            sig = (0.4 * tone + 0.55 * whir) * amp
            sig = lowpass(highpass(sig, 80, 2), 5500, 4)
            gp = np.array([self.persp.gain_pan(self._source_pos(g, x), x) for x in np.arange(0, self.dur, 0.05)])
            gx = np.arange(0, self.dur, 0.05)
            gain_c = np.interp(tt, gx, gp[:, 0]) * gain
            pan_c = np.interp(tt, gx, gp[:, 1])
            self.mix.place_stream(sig, gain_c, pan_c, wet=0.25)

    def _source_pos(self, group, t):
        if group.startswith("arm") or group.startswith("fing"):
            side = group[-1]
            return self.c.hand(side, t)[0].translation
        if group == "head":
            return (0.0, -0.02, 1.38)
        return (0.0, 0.02, 1.0)

    def electronics(self):
        """Power-on relay + idle drive hum, cooling fan while computing/solving."""
        T = self.c.T
        n = self.mix.n
        tt = np.arange(n) / SR
        r = self.rng
        self.mix.stem = "electronics"
        relay = modal(r, freqs=[1300, 2650, 4100], taus=[0.006, 0.004, 0.0025], amps=[1, 0.7, 0.4], dur=0.05, noise=0.5, noise_ms=0.4)
        g, pan = self.persp.gain_pan((0, 0.12, 1.06), T["eyes_on"])
        self.mix.place(relay, T["eyes_on"] - 0.06, 0.07 * g, pan, wet=0.15)
        self.events.append((T["eyes_on"] - 0.06, "relay", 1))
        # idle drive hum (switching regulators / holding torque)
        on = np.clip((tt - T["eyes_on"]) / 0.6, 0, 1)
        am = 1 + 0.25 * np.sin(2 * np.pi * 0.7 * tt) + 0.15 * np.sin(2 * np.pi * 3.1 * tt)
        hum = (np.sin(2 * np.pi * 1190 * tt + 0.8 * np.sin(2 * np.pi * 0.3 * tt)) + 0.3 * np.sin(2 * np.pi * 2380 * tt)) * on * am
        hum += 0.6 * bandpass(r.normal(0, 1, n), 4000, 9000, 2) * on * 0.25
        gp = np.array([self.persp.gain_pan((0, 0.05, 1.05), x) for x in np.arange(0, self.dur, 0.05)])
        gx = np.arange(0, self.dur, 0.05)
        self.mix.place_stream(hum, np.interp(tt, gx, gp[:, 0]) * db(-70), np.interp(tt, gx, gp[:, 1]), wet=0.2)
        # fan: spins up while thinking, holds during the solve, spins down after
        t_up, t_dn = T["think"][0], self.c.solve.t_end + 1.2
        rpm = np.clip((tt - t_up) / 1.1, 0, 1) * np.clip(1 - (tt - t_dn) / 2.0, 0, 1)
        rpm = rpm ** 1.5
        bpf = 190 * rpm + 1e-3
        ph = 2 * np.pi * np.cumsum(bpf) / SR
        fan = 0.5 * np.sin(ph) + 0.25 * np.sin(2 * ph) + 1.2 * lowpass(bandpass(r.normal(0, 1, n), 200, 6000, 2), 3500, 2) * rpm
        fan *= rpm
        gp = np.array([self.persp.gain_pan((0, 0.15, 1.05), x) for x in np.arange(0, self.dur, 0.05)])
        self.mix.place_stream(fan, np.interp(tt, gx, gp[:, 0]) * db(-41), np.interp(tt, gx, gp[:, 1]) * 0.5, wet=0.25)

    def room_tone(self):
        n = self.mix.n
        r = self.rng
        tt = np.arange(n) / SR
        for ch in range(2):
            hvac = lowpass(np.cumsum(r.normal(0, 1, n)) * 0.02, 380, 2)
            hvac = highpass(hvac, 35, 2)
            hvac /= np.std(hvac) + 1e-9
            air = bandpass(r.normal(0, 1, n), 120, 2400, 2)
            air /= np.std(air)
            city = lowpass(r.normal(0, 1, n), 160, 4)
            city /= np.std(city)
            swell = 1 + 0.5 * np.sin(2 * np.pi * 0.07 * tt + ch) + 0.3 * np.sin(2 * np.pi * 0.19 * tt)
            # one distant vehicle passing outside (through glass: low-passed)
            car = lowpass(r.normal(0, 1, n), 700, 2)
            car /= np.std(car)
            car_env = np.exp(-((tt - 8.6) / 1.8) ** 2) * (0.8 if ch else 1.0)
            # 100 Hz electrical hum from the LED drivers (barely there)
            mains = np.sin(2 * np.pi * 100 * tt + ch * 0.4) + 0.3 * np.sin(2 * np.pi * 300 * tt)
            x = db(-64) * hvac + db(-71) * air + db(-67) * city * swell + db(-68) * car * car_env + db(-82) * mains
            self.mix.amb[ch] += db(-7) * x
        self.mix.stems["room_tone"] = self.mix.amb.copy()

    # ---------------------------------------------------------------- render
    def render(self, rig, verbose=True):
        self.room_tone()
        self.electronics()
        self.handling()
        self.cube_turns()
        self.servos(rig)
        ir = make_ir(self.rng)
        wet = np.array([signal.fftconvolve(self.mix.send[ch], ir[ch])[: self.mix.n] for ch in range(2)])
        out = self.mix.dry + wet + self.mix.amb
        out = highpass(out, 22, 2)
        return out

    @staticmethod
    def master(x, target_lufs=-23.0, ceiling_dbtp=-1.5, comp_thresh_db=-24.0, comp_ratio=2.0, return_gain=False):
        """Film-style master: gentle program compression (so the dense solve does not jump
        far above the rest of the film), BS.1770 loudness normalisation, true-peak limiting.

        Returns the mastered signal (and, with return_gain, the per-sample gain applied,
        so stems can be scaled exactly like the mix)."""
        g0 = db(target_lufs - integrated_lufs(x))
        comp = _compressor_gain(x * g0, comp_thresh_db, comp_ratio)
        y = x * g0 * comp
        g1 = db(target_lufs - integrated_lufs(y))
        y = y * g1
        # look-ahead brick-wall limiter on 4x oversampled peaks
        ceil = db(ceiling_dbtp)
        up = signal.resample_poly(y, 4, 1, axis=1)
        peak = np.max(np.abs(up), axis=0)
        peak = np.maximum.reduceat(peak, np.arange(0, len(peak), 4))[: y.shape[1]]
        gr = np.minimum(1.0, ceil / np.maximum(peak, 1e-9))
        # attack 1.5 ms look-ahead, release 80 ms
        la = int(0.0015 * SR)
        gr_la = _min_filter(gr, la)
        lim = np.empty_like(gr_la)
        cur = 1.0
        rel = math.exp(-1 / (0.08 * SR))
        for i, v in enumerate(gr_la):
            cur = v if v < cur else v + (cur - v) * rel
            lim[i] = cur
        gain = g0 * g1 * comp * lim
        return (x * gain, gain) if return_gain else x * gain


def _compressor_gain(x, thresh_db, ratio, win_ms=20.0, attack_ms=5.0, release_ms=180.0, knee_db=6.0):
    """Feed-forward RMS compressor (soft knee), gain computed on a 1 ms control grid."""
    hop = SR // 1000
    p = np.mean(x ** 2, axis=0)
    w = int(win_ms / 1000 * SR)
    p = np.convolve(p, np.ones(w) / w, mode="same")[::hop]
    lvl = 10 * np.log10(p + 1e-14)
    over = lvl - thresh_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over, (over + knee_db / 2) ** 2 / (2 * knee_db)))
    target = -gr * (1 - 1 / ratio)  # dB, <= 0
    a = math.exp(-1 / attack_ms)
    r = math.exp(-1 / release_ms)
    g = np.empty_like(target)
    cur = 0.0
    for i, v in enumerate(target):
        cur = v + (cur - v) * (a if v < cur else r)
        g[i] = cur
    n = x.shape[1]
    return db(np.interp(np.arange(n), np.arange(len(g)) * hop, g))


def _attack_release(x, attack_s, release_s):
    """One-pole envelope follower with separate rise and fall times."""
    a = math.exp(-1 / (attack_s * SR))
    r = math.exp(-1 / (release_s * SR))
    out = np.empty_like(x)
    cur = 0.0
    for i, v in enumerate(x):
        cur = v + (cur - v) * (a if v > cur else r)
        out[i] = cur
    return out


def _min_filter(x, half):
    from scipy.ndimage import minimum_filter1d

    return minimum_filter1d(x, size=2 * half + 1, mode="nearest")


# ----------------------------------------------------------------------------- loudness
def k_weight(x):
    # ITU-R BS.1770 K-weighting (48 kHz coefficients)
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)


def integrated_lufs(x):
    y = k_weight(x)
    blk = int(0.4 * SR)
    hop = int(0.1 * SR)
    ms = []
    for i in range(0, y.shape[1] - blk, hop):
        ms.append(np.sum(np.mean(y[:, i : i + blk] ** 2, axis=1)))
    ms = np.array(ms)
    lk = -0.691 + 10 * np.log10(ms + 1e-20)
    g = ms[lk > -70]
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 10
    g2 = ms[(lk > -70) & (lk > rel)]
    return -0.691 + 10 * np.log10(np.mean(g2))


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    data = np.round(x.T * 8388607).astype("<i4")  # interleaved L R
    raw = data.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(raw)

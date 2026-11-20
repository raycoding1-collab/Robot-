"""Small physical models used by the choreography.

* Layer dynamics of a magnetic speed cube (finger flick -> free coasting with
  Coulomb + viscous friction -> magnetic detent capture with damped overshoot).
* Compliant-grip reaction ("the whole cube recoils a little when a layer is
  accelerated").
* Critically damped follower used for head/torso secondary motion.
"""
from __future__ import annotations

import math
from bisect import bisect_left

QUARTER = math.pi / 2


def min_jerk(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * s * (10 - 15 * s + 6 * s * s)


def min_jerk_d(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return 30 * s * s * (1 - s) ** 2


class Profile:
    """Sampled angle(t) for one quarter turn, t relative to its start."""

    def __init__(self, ts, th):
        self.ts = ts
        self.th = th
        self.duration = ts[-1]

    def __call__(self, t):
        if t <= 0:
            return 0.0
        if t >= self.duration:
            return self.th[-1]
        i = bisect_left(self.ts, t)
        t0, t1 = self.ts[i - 1], self.ts[i]
        a = (t - t0) / (t1 - t0)
        return self.th[i - 1] * (1 - a) + self.th[i] * a

    def accel(self, t, h=1e-3):
        return (self(t + h) - 2 * self(t) + self(t - h)) / (h * h)


def hand_driven_profile(T: float, dt=1e-3) -> Profile:
    n = int(math.ceil(T / dt))
    ts = [i * T / n for i in range(n + 1)]
    th = [QUARTER * min_jerk(t / T) for t in ts]
    return Profile(ts, th)


# ---------------------------------------------------------------- flick model
DETENT_K = 42000.0  # rad/s^2 per rad near alignment (magnets)
DETENT_W = math.radians(8.0)  # magnet capture width
COULOMB = 600.0  # rad/s^2 core/spring friction
VISCOUS = 14.0  # 1/s
DT = 1e-4


def _detent(theta):
    d = theta - QUARTER * round(theta / QUARTER)
    return -DETENT_K * d * math.exp(-((d / DETENT_W) ** 2))


def _simulate_flick(theta_rel, w_rel, t_max=0.25):
    """Constant-acceleration finger drive to (theta_rel, w_rel), then free layer."""
    T_d = 2 * theta_rel / w_rel
    a = w_rel / T_d
    ts, th = [], []
    t = 0.0
    while t < T_d:
        ts.append(t)
        th.append(0.5 * a * t * t)
        t += DT
    theta, w = theta_rel, w_rel
    peak = theta
    settled_for = 0.0
    while t < T_d + t_max:
        dd = theta - QUARTER * round(theta / QUARTER)
        visc = VISCOUS + (55.0 if abs(dd) < DETENT_W else 0.0)
        acc = _detent(theta) - visc * w - (COULOMB * math.copysign(1.0, w) if abs(w) > 1e-3 else 0.0)
        # static friction holds the layer once it is slow and the magnet is weak
        if abs(w) < 1e-3 and abs(_detent(theta)) < COULOMB:
            acc = 0.0
            w = 0.0
        w += acc * DT
        theta += w * DT
        peak = max(peak, theta)
        ts.append(t)
        th.append(theta)
        if w == 0.0 or (abs(theta - QUARTER) < math.radians(0.3) and abs(w) < 0.3):
            settled_for += DT
            if settled_for > 0.004:
                break
        else:
            settled_for = 0.0
        t += DT
    return ts, th, peak, T_d


def flick_profile(theta_rel_deg=40.0, margin=1.05) -> tuple[Profile, float]:
    """Smallest release speed that lets the magnets capture the layer at 90deg,
    plus a small safety margin (a precise robot does not over-flick)."""
    theta_rel = math.radians(theta_rel_deg)
    lo, hi = 5.0, 120.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ts, th, _, _ = _simulate_flick(theta_rel, mid)
        if th[-1] < QUARTER - math.radians(5):
            lo = mid
        else:
            hi = mid
    ts, th, peak, T_d = _simulate_flick(theta_rel, hi * margin)
    # magnets pull the last fraction of a degree home over ~12 ms
    err = th[-1] - QUARTER
    t = ts[-1]
    for k in range(1, 121):
        ts.append(t + k * DT)
        th.append(QUARTER + err * (1 - min_jerk(k / 120)))
    # decimate to 0.5 ms
    ts2, th2 = ts[::5], th[::5]
    if ts2[-1] != ts[-1]:
        ts2.append(ts[-1])
        th2.append(QUARTER)
    return Profile(ts2, th2), T_d


# ------------------------------------------------------------ grip compliance
def grip_response(times, torque_fn, omega=70.0, zeta=0.55, gain=0.028, dt=5e-4):
    """Integrate phi'' = -gain*torque(t) - 2 zeta w phi' - w^2 phi on a grid."""
    t0, t1 = times[0], times[-1]
    n = int((t1 - t0) / dt) + 2
    ts, ph = [], []
    phi = v = 0.0
    t = t0
    for _ in range(n):
        ts.append(t)
        ph.append(phi)
        a = -gain * torque_fn(t) - 2 * zeta * omega * v - omega * omega * phi
        v += a * dt
        phi += v * dt
        t += dt
    return Profile([x - t0 for x in ts], ph), t0


class Smoother:
    """Critically damped 2nd-order follower (natural frequency w)."""

    def __init__(self, w):
        self.w = w

    def run(self, ts, xs):
        out = []
        x = list(xs[0]) if isinstance(xs[0], (list, tuple)) else xs[0]
        v = [0.0] * len(x) if isinstance(x, list) else 0.0
        prev = ts[0]
        for t, target in zip(ts, xs):
            dt = t - prev
            prev = t
            if isinstance(x, list):
                for k in range(len(x)):
                    # sub-step for stability
                    steps = max(1, int(dt * self.w / 0.2))
                    h = dt / steps
                    for _ in range(steps):
                        a = self.w * self.w * (target[k] - x[k]) - 2 * self.w * v[k]
                        v[k] += a * h
                        x[k] += v[k] * h
                out.append(tuple(x))
            else:
                steps = max(1, int(dt * self.w / 0.2))
                h = dt / steps
                for _ in range(steps):
                    a = self.w * self.w * (target - x) - 2 * self.w * v
                    v += a * h
                    x += v * h
                out.append(x)
        return out

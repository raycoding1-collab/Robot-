"""Numerically tune the hand grip frames (run with the Blender python).

Scores a grip by: fingertip reach residuals, finger/palm penetration of the cube
(and the table for the pick grip), and joint naturalness, over all the
configurations the solve uses (rest grip, wrist turns, thumb/middle twists,
index/ring flicks). Prints the optimised parameters.
"""
import math
import random
import sys

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
import bpy  # noqa: F401  (provides mathutils)
from mathutils import Matrix, Vector

from robot_rubiks import ik
from robot_rubiks.choreography import CONTACT, HOVER, RETRACT, SURF, hand_frame
from robot_rubiks.rig_def import FINGERS

QUART = math.pi / 2


def rot(axis, a):
    return Matrix.Rotation(a, 4, "XYZ"[axis])


def box_sdf(p, h=SURF):
    q = Vector((abs(p.x) - h, abs(p.y) - h, abs(p.z) - h))
    outside = Vector((max(q.x, 0), max(q.y, 0), max(q.z, 0))).length
    inside = min(max(q.x, q.y, q.z), 0.0)
    return outside + inside


def finger_points(fn, psi, q1, q2):
    """Sample points (canonical hand space) along segments with radii."""
    f = FINGERS[fn]
    q_1, q_2, q_3 = ik.finger_locals(fn, psi, q1, q2)
    m = Matrix.Translation(f["base"]) @ q_1.to_matrix().to_4x4()
    pts = []
    lens = f["lens"]
    r = f["thick"] * 0.5
    for k, (L, q) in enumerate(zip(lens, (None, q_2, q_3))):
        if k == 1:
            m = m @ Matrix.Translation((0, lens[0], 0)) @ q_2.to_matrix().to_4x4()
        if k == 2:
            m = m @ Matrix.Translation((0, lens[1], 0)) @ q_3.to_matrix().to_4x4()
        n = 4
        for i in range(n + 1):
            s = i / n
            if k == 2 and s > 0.45:
                continue
            pts.append((m @ Vector((0, s * L, 0)), r * (1 - 0.1 * k), k))
    return pts


def palm_points():
    pts = []
    for x in (-0.03, -0.01, 0.01, 0.03):
        for y in (0.02, 0.05, 0.08):
            pts.append(Vector((x, y, -0.016)))
    return pts


def evaluate(G, configs, table=False, verbose=False):
    cost = 0.0
    report = {}
    for name, palm_rot, targets in configs:
        palm = palm_rot @ G
        inv = palm.inverted()
        for fn, tgt in targets.items():
            q = inv @ tgt
            ang = ik.finger_ik(fn, q)
            pad = ik.finger_pad_world(fn, *ang)
            err = (pad - q).length
            cost += (err * 1000) ** 2 * 4
            report[(name, fn)] = round(err * 1000, 1)
            psi, q1, q2 = ang
            if fn == "thumb":
                cost += 30 * max(0, abs(psi) - 0.55) ** 2 + 30 * max(0, -q1 - 0.1) ** 2
            else:
                cost += 60 * max(0, abs(psi) - 0.2) ** 2 + 40 * max(0, -q1) ** 2 + 20 * max(0, q2 - 1.5) ** 2
            for p, r, k in finger_points(fn, *ang):
                pw = palm @ p
                d = box_sdf(pw) - r * 0.85
                if d < 0:
                    cost += (d * 1000) ** 2 * 3
                if table:
                    dt = (pw.z + SURF) - r
                    if dt < 0:
                        cost += (dt * 1000) ** 2 * 3
        # passive fingers (pinky) + palm
        for p in palm_points():
            pw = palm @ p
            d = box_sdf(pw) - 0.004
            if d < 0:
                cost += (d * 1000) ** 2 * 3
            if table and pw.z + SURF < 0.006:
                cost += ((pw.z + SURF - 0.006) * 1000) ** 2 * 3
        for p, r, k in finger_points("pinky", 0.10, 0.95, 1.25):
            pw = palm @ p
            d = box_sdf(pw) - r
            if d < 0:
                cost += (d * 1000) ** 2 * 3
            if table and pw.z + SURF - r < 0:
                cost += ((pw.z + SURF - r) * 1000) ** 2 * 3
        # forearm must come from the robot side (-Y in H coords) and below-ish
        yh = palm.col[1].to_3d()
        cost += 200 * max(0, 0.55 - yh.y) ** 2
        w = palm.translation
        cost += 2000 * max(0, w.z - 0.02) ** 2
    if verbose:
        return cost, report
    return cost


def solve_configs(alpha=0.62, beta=0.08):
    C = {k: v[0] for k, v in CONTACT.items()}
    home = {"thumb": C["thumb"], "middle": C["middle"], "index": HOVER["index"], "ring": HOVER["ring"]}
    cfg = [("home", Matrix.Identity(4), dict(home))]
    for a in (-QUART, -QUART / 2, QUART / 2, QUART):
        cfg.append((f"wrist{a:.2f}", rot(0, alpha * a), {k: rot(0, a) @ v for k, v in home.items()}))
    for a in (-math.radians(42), math.radians(42)):
        t = dict(home)
        t["thumb"] = rot(1, a) @ C["thumb"]
        cfg.append((f"thumbtw{a:.2f}", rot(1, beta * a), t))
        t = dict(home)
        t["middle"] = rot(1, a) @ C["middle"]
        t["index"] = RETRACT["index"]
        t["ring"] = RETRACT["ring"]
        cfg.append((f"midtw{a:.2f}", rot(1, beta * a), t))
    for a in (0.0, -math.radians(40)):
        t = dict(home)
        t["index"] = rot(2, a) @ C["index"]
        cfg.append((f"flickU{a:.2f}", Matrix.Identity(4), t))
        t = dict(home)
        t["ring"] = rot(2, a) @ C["ring"]
        cfg.append((f"flickD{a:.2f}", Matrix.Identity(4), t))
    return cfg


def pick_configs():
    C = {k: v[0] for k, v in CONTACT.items()}
    t = {"thumb": C["thumb"], "middle": C["middle"], "index": RETRACT["index"], "ring": RETRACT["ring"] + Vector((0, 0, 0.03))}
    return [("pick", Matrix.Identity(4), t)]


def make(params):
    ox, oy, oz, rx, ry, rz = params
    base = hand_frame((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (ox, oy, oz))
    r = Matrix.Rotation(rz, 4, "Z") @ Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rx, 4, "X")
    loc = base.translation.copy()
    m = r @ base.to_3x3().to_4x4()
    m.translation = loc
    return m


def optimise(configs, x0, table=False, iters=6000, seed=1):
    """configs: callable(alpha, beta) -> list, x0: 8 params."""
    rng = random.Random(seed)
    best = list(x0)
    def f(x):
        return evaluate(make(x[:6]), configs(x[6], x[7]), table) + 50 * max(0, x[6] - 0.9) ** 2 + 50 * max(0, 0.3 - x[6]) ** 2
    fb = f(best)
    steps = [0.006, 0.006, 0.006, 0.08, 0.08, 0.08, 0.05, 0.05]
    for it in range(iters):
        scale = 1.0 - 0.9 * it / iters
        cand = [b + rng.gauss(0, s * scale) for b, s in zip(best, steps)]
        fc = f(cand)
        if fc < fb:
            best, fb = cand, fc
    return best, fb


if __name__ == "__main__":
    results = []
    for seed, x0 in enumerate([
        [0.0827, -0.1009, -0.0457, 0.2236, 0.116, 0.2512, 0.62, 0.08],
        [0.056, -0.085, -0.004, 0.0, 0.0, 0.0, 0.75, 0.2],
        [0.07, -0.09, -0.03, 0.2, 0.3, 0.2, 0.8, 0.3],
    ]):
        best, f = optimise(solve_configs, x0, iters=5000, seed=seed + 3)
        results.append((f, best))
        print("restart", seed, round(f, 1))
    f, best = min(results)
    c, rep = evaluate(make(best[:6]), solve_configs(best[6], best[7]), verbose=True)
    print("SOLVE params", [round(v, 4) for v in best], "cost", round(c, 1))
    print({f"{k[0]}:{k[1]}": v for k, v in rep.items() if v > 1.0})
    best2, f2 = optimise(lambda a, b: pick_configs(), best, table=True, iters=3000)
    c2, rep2 = evaluate(make(best2[:6]), pick_configs(), table=True, verbose=True)
    print("PICK params", [round(v, 4) for v in best2], "cost", round(c2, 1))
    print(rep2)

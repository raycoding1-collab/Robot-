"""Time-continuous choreography: cube, hands, fingers, head, torso.

Everything is an analytic / simulated function of time `t` (seconds) so the
animation can be sampled at any (sub)frame for exact motion blur.

Conventions
-----------
H(t)       world matrix of the "holding frame" (cube-local axes when held upright)
Q(t)       rotation of the cube body relative to H (inspection rotations)
grips      hand frames expressed in H coordinates (canonical right hand, left mirrored)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from mathutils import Matrix, Quaternion, Vector

from . import physics as P
from .cube_logic import CubeState, Move, find_scramble_and_solution
from .rig_def import CAM_END_POS, CUBE_HALF, CUBE_REST_POS, CUBE_UPRIGHT, MIRROR, TABLE_TOP_Z

S3 = Matrix.Diagonal((-1.0, 1.0, 1.0))
S4 = MIRROR
AX = [Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))]
SURF = CUBE_HALF + 0.0004 + 0.0001  # tile top surface
PAD_EMBED = 0.00025  # rubber pads compress slightly on contact
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def rot(axis: int, ang: float) -> Matrix:
    return Matrix.Rotation(ang, 4, "XYZ"[axis])


def lerp(a, b, s):
    return a + (b - a) * s


def smooth(s):
    return P.min_jerk(s)


def mat_lerp(A: Matrix, B: Matrix, s: float) -> Matrix:
    la, ra, _ = A.decompose()
    lb, rb, _ = B.decompose()
    m = ra.slerp(rb, s).to_matrix().to_4x4()
    m.translation = la.lerp(lb, s)
    return m


def mirror_m(m: Matrix) -> Matrix:
    return S4 @ m @ S4


def mirror_v(v: Vector) -> Vector:
    return Vector((-v.x, v.y, v.z))


def hand_frame(x, y, origin):
    x = Vector(x).normalized()
    y = Vector(y)
    y = (y - x * y.dot(x)).normalized()
    z = x.cross(y)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = Vector(origin)
    return m


# ----------------------------------------------------------------------------- grips
# Right hand C-grip on the R side of the cube: palm faces the R face, fingers wrap
# around the back (B) face, thumb on the front (F) face. Rows of the back face line
# up with index (U row) / middle (E row) / ring (D row).
def grip_from_params(params):
    """Grip frame from (origin xyz, rotations about H x/y/z) — see tools/tune_grips.py."""
    ox, oy, oz, rx, ry, rz = params
    base = hand_frame((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (ox, oy, oz))
    r = Matrix.Rotation(rz, 4, "Z") @ Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rx, 4, "X")
    m = r @ base.to_3x3().to_4x4()
    m.translation = Vector((ox, oy, oz))
    return m


# optimised with tools/tune_grips.py (reach residual <~2 mm, no finger/cube/table penetration)
SOLVE_GRIP_PARAMS = (0.0881, -0.1152, -0.0278, 0.0784, 0.0411, 0.2457)
WRIST_PALM_RATIO = 0.775  # share of a wrist turn done by the palm (rest by the fingers)
TWIST_PALM_RATIO = 0.1439
PICK_GRIP_PARAMS = (0.0985, -0.1059, 0.0089, 0.064, -0.1959, 0.2218)
G_SOLVE_R = grip_from_params(SOLVE_GRIP_PARAMS)
G_PICK_R = grip_from_params(PICK_GRIP_PARAMS)
G_INSPECT_R = G_SOLVE_R


def grip_L(g):
    return S4 @ g @ S4


# contact points (H coords, right hand) and their surface normals
CONTACT = {
    "thumb": (Vector((0.019, -SURF + PAD_EMBED, 0.0)), Vector((0, -1, 0))),
    "middle": (Vector((0.019, SURF - PAD_EMBED, 0.0)), Vector((0, 1, 0))),
    "index": (Vector((0.019, SURF - PAD_EMBED, 0.019)), Vector((0, 1, 0))),
    "ring": (Vector((0.019, SURF - PAD_EMBED, -0.019)), Vector((0, 1, 0))),
}
HOVER = {
    "index": Vector((0.0215, SURF + 0.0035, 0.0205)),
    "ring": Vector((0.0215, SURF + 0.0035, -0.0205)),
}
RETRACT = {
    "index": Vector((0.025, SURF + 0.0065, 0.027)),
    "ring": Vector((0.025, SURF + 0.0065, -0.027)),
}

# relaxed finger poses (psi, q1, q2)
POSE_OPEN = {"thumb": (0.25, 0.05, 0.10), "index": (0.0, 0.10, 0.15), "middle": (0.0, 0.10, 0.15), "ring": (0.0, 0.12, 0.18), "pinky": (0.05, 0.15, 0.2)}
POSE_CURL_PINKY = (0.10, 0.95, 1.25)
POSE_RELAX = {"thumb": (0.35, 0.15, 0.25), "index": (0.0, 0.35, 0.45), "middle": (0.0, 0.40, 0.50), "ring": (0.0, 0.45, 0.55), "pinky": (0.05, 0.5, 0.6)}


def square_radius(phi):
    """Radius of the cube's square cross-section boundary at polar angle phi."""
    c = max(abs(math.cos(phi)), abs(math.sin(phi)))
    return SURF / c


def polar(axis, p: Vector):
    """(axial, radius, angle) of point p around a principal axis through the centre."""
    a, b = [(1, 2), (2, 0), (0, 1)][axis]
    return p[axis], math.hypot(p[a], p[b]), math.atan2(p[b], p[a])


def from_polar(axis, ax_v, r, ang):
    a, b = [(1, 2), (2, 0), (0, 1)][axis]
    v = Vector((0, 0, 0))
    v[axis] = ax_v
    v[a] = r * math.cos(ang)
    v[b] = r * math.sin(ang)
    return v


def regrip_path(axis, p_home: Vector, ang_from: float, s: float, clearance=0.007):
    """Fingertip path that unwinds `ang_from` -> 0 around `axis` while clearing the cube."""
    ax_v, r0, a0 = polar(axis, p_home)
    a = a0 + ang_from * (1 - smooth(s))
    excess = r0 - square_radius(a0)  # pads embed a hair; keep that offset
    r = square_radius(a) + excess + clearance * math.sin(math.pi * min(1.0, max(0.0, s))) ** 0.6
    # fingertips sitting off the mid-plane are fine with the same rule
    return from_polar(axis, ax_v, r, a)


# ----------------------------------------------------------------------------- turns
@dataclass
class Turn:
    move: Move  # quarter-turn move (turns = +-1)
    kind: str  # wrist | flick | twist
    side: str  # R | L | both
    finger: str | None
    t0: float = 0.0
    prof: P.Profile | None = None
    t_release: float = 0.0  # flick: finger leaves the cubie
    regrip: float = 0.0
    members: list = field(default_factory=list)
    R_before: list = field(default_factory=list)

    @property
    def axis(self):
        return self.move.axis

    @property
    def sgn(self):
        return 1 if self.move.signed_axis_angle_quarters > 0 else -1

    @property
    def t_end(self):
        return self.t0 + self.prof.duration

    @property
    def t_free(self):
        return self.t_end + self.regrip

    def angle(self, t):
        return self.sgn * self.prof(t - self.t0)


def classify(m: Move):
    f, d = m.face, m.turns
    if f == "R":
        return "wrist", "R", None
    if f == "L":
        return "wrist", "L", None
    if f == "U":
        return ("flick", "R", "index") if d == 1 else ("flick", "L", "index")
    if f == "D":
        return ("flick", "L", "ring") if d == 1 else ("flick", "R", "ring")
    if f == "F":
        return "twist", "both", "thumb"
    if f == "B":
        return "twist", "both", "middle"
    raise ValueError(m)


class Solve:
    """Schedules a face-turn solution onto the robot's effectors."""

    WRIST_T = 0.098
    WRIST_REGRIP = 0.068
    TWIST_T = 0.104
    TWIST_REGRIP = 0.066
    FLICK_APPROACH = 0.028
    FLICK_RETURN = 0.062
    GAP = 0.008

    def __init__(self, state: CubeState, solution: list[Move], t_start: float, seed=3):
        rng = random.Random(seed)
        self.turns: list[Turn] = []
        flick, t_drive = P.flick_profile()
        self.flick_prof = flick
        st = state.copy()
        busy = {}  # effector -> time free

        def eff_of(kind, side, finger):
            if kind == "wrist":
                return [f"{side}:{f}" for f in FINGERS] + [f"{side}:palm"]
            if kind == "flick":
                return [f"{side}:{finger}"]
            return [f"R:{finger}", f"L:{finger}"]

        def holders(kind, side, finger):
            # effectors that must be gripping (not moving) for this turn
            if kind == "wrist":
                o = "L" if side == "R" else "R"
                return [f"{o}:thumb", f"{o}:middle", f"{o}:palm"]
            if kind == "flick":
                return ["R:thumb", "R:middle", "L:thumb", "L:middle", "R:palm", "L:palm"]
            other = "middle" if finger == "thumb" else "thumb"
            extra = ["R:index", "L:index", "R:ring", "L:ring"] if finger == "middle" else []
            return [f"R:{other}", f"L:{other}", "R:palm", "L:palm"] + extra

        t_prev_end = t_start
        for m in solution:
            reps = 2 if m.turns == 2 else 1
            qm = Move(m.face, 1 if m.turns == 2 else m.turns)
            for _ in range(reps):
                kind, side, finger = classify(qm)
                tr = Turn(qm, kind, side, finger)
                if kind == "wrist":
                    tr.prof = P.hand_driven_profile(self.WRIST_T * rng.uniform(0.94, 1.06))
                    tr.regrip = self.WRIST_REGRIP
                elif kind == "twist":
                    # thumbs / middle fingers flick the layer like a speedcuber; magnets finish it
                    tr.prof = flick
                    tr.regrip = self.TWIST_REGRIP
                else:
                    tr.prof = flick
                    tr.regrip = self.FLICK_RETURN
                effs = eff_of(kind, side, finger)
                need = effs + holders(kind, side, finger)
                t0 = max([t_prev_end + self.GAP] + [busy.get(e, 0.0) for e in need])
                if kind == "flick":
                    t0 = max(t0, max(busy.get(e, 0.0) for e in effs) + self.FLICK_APPROACH)
                if kind in ("flick", "twist"):
                    tr.t_release = t0 + t_drive
                tr.t0 = t0
                members = st.layer_members(qm)
                tr.members = members
                tr.R_before = [r.copy() for r in st.R]
                st.apply(qm)
                for e in effs:
                    if kind == "wrist":
                        busy[e] = tr.t_free
                    elif kind == "twist":
                        busy[e] = tr.t_release + self.TWIST_REGRIP
                    else:
                        busy[e] = tr.t_release + self.FLICK_RETURN
                # holders are not blocked by a flick's return stroke
                t_prev_end = tr.t_end
                self.turns.append(tr)
        assert st.is_solved()
        self.final_R = [r.copy() for r in st.R]
        self.t_start = t_start
        self.t_end = max(tr.t_free for tr in self.turns)
        self.t_last_settle = max(tr.t_end for tr in self.turns)
        # reaction of the compliant grip to layer accelerations, per axis
        self.recoil = []
        for axis in range(3):
            def torque(t, axis=axis):
                tot = 0.0
                for tr in self.turns:
                    if tr.axis == axis and tr.t0 - 0.01 <= t <= tr.t_end + 0.01:
                        tot += tr.sgn * tr.prof.accel(t - tr.t0)
                return tot

            prof, t0 = P.grip_response([t_start - 0.05, self.t_end + 0.6], torque)
            self.recoil.append((prof, t0))

    def active(self, t):
        for tr in self.turns:
            if tr.t0 <= t < tr.t_end:
                return tr
        return None

    def cubie_rotations(self, t, R_init):
        """Integer state + live layer rotation -> list of 4x4 rotations (cube-local)."""
        prev = None
        for tr in self.turns:
            if t < tr.t0:
                break
            prev = tr
        if prev is None:
            return [Matrix([list(map(float, row)) for row in r]).to_4x4() for r in R_init]
        if t < prev.t_end:
            base = prev.R_before
            ang = prev.angle(t)
            Rl = rot(prev.axis, ang)
            out = []
            mem = set(prev.members)
            for i, r in enumerate(base):
                m = Matrix([list(map(float, row)) for row in r]).to_4x4()
                out.append(Rl @ m if i in mem else m)
            return out
        # completed prev -> state after prev
        idx = self.turns.index(prev)
        if idx + 1 < len(self.turns):
            after = self.turns[idx + 1].R_before
        else:
            after = self.final_R
        return [Matrix([list(map(float, row)) for row in r]).to_4x4() for r in after]

    def recoil_matrix(self, t):
        m = Matrix.Identity(4)
        for axis, (prof, t0) in enumerate(self.recoil):
            m = m @ rot(axis, prof(t - t0))
        return m

    # ---------------------------------------------------------------- effectors
    def hand_state(self, side, t):
        """Palm frame (H coords) and finger targets (H coords) during the solve."""
        g = G_SOLVE_R if side == "R" else grip_L(G_SOLVE_R)
        mir = side == "L"

        def C(name):
            p, n = CONTACT[name]
            return (mirror_v(p), mirror_v(n)) if mir else (p, n)

        def V(dic, name):
            return mirror_v(dic[name]) if mir else dic[name].copy()

        targets = {
            "thumb": C("thumb")[0].copy(),
            "middle": C("middle")[0].copy(),
            "index": V(HOVER, "index"),
            "ring": V(HOVER, "ring"),
        }
        palm = g.copy()
        for tr in self.turns:
            if t < tr.t0 - 0.06 or t > max(tr.t_free, tr.t_release + tr.regrip) + 0.04:
                continue
            ax = tr.axis
            if tr.kind == "wrist" and tr.side == side:
                if tr.t0 <= t < tr.t_end:
                    a = tr.angle(t)
                    palm = rot(ax, WRIST_PALM_RATIO * a) @ g
                    for k in targets:
                        targets[k] = rot(ax, a) @ targets[k]
                elif tr.t_end <= t < tr.t_free:
                    s = (t - tr.t_end) / tr.regrip
                    a_end = tr.sgn * P.QUARTER
                    palm = rot(ax, WRIST_PALM_RATIO * a_end * (1 - smooth(s))) @ g
                    for k in targets:
                        clr = 0.007 if k in ("thumb", "middle") else 0.010
                        targets[k] = regrip_path(ax, targets[k], a_end, s, clr)
            elif tr.kind == "twist":
                fn = tr.finger
                t_ret_ = tr.t_release + tr.regrip
                if fn == "middle" and tr.t0 - 0.05 <= t < t_ret_ + 0.03:
                    # keep index/ring clear of the back layer
                    w = smooth(min(1.0, (t - (tr.t0 - 0.05)) / 0.05)) * (1 - smooth(max(0.0, (t - t_ret_) / 0.03)))
                    for k in ("index", "ring"):
                        targets[k] = targets[k].lerp(V(RETRACT, k), w)
                t_ret = tr.t_release + tr.regrip
                if tr.t0 <= t < tr.t_release:
                    a = tr.angle(t)
                    targets[fn] = rot(ax, a) @ targets[fn]
                    palm = rot(ax, TWIST_PALM_RATIO * a) @ palm
                elif tr.t_release <= t < t_ret:
                    s = (t - tr.t_release) / tr.regrip
                    a_rel = tr.angle(tr.t_release)
                    home = targets[fn]
                    ang = a_rel * (1 - smooth(s))
                    p = rot(ax, ang) @ home
                    n = C(fn)[1]
                    targets[fn] = p + n * (0.0045 * math.sin(math.pi * s))
                    palm = rot(ax, TWIST_PALM_RATIO * ang) @ palm
            elif tr.kind == "flick" and tr.side == side:
                fn = tr.finger
                contact = C(fn)[0]
                hov = targets[fn]
                if tr.t0 - self.FLICK_APPROACH <= t < tr.t0:
                    s = (t - (tr.t0 - self.FLICK_APPROACH)) / self.FLICK_APPROACH
                    targets[fn] = hov.lerp(contact, smooth(s))
                elif tr.t0 <= t < tr.t_release:
                    targets[fn] = rot(ax, tr.angle(t)) @ contact
                elif tr.t_release <= t < tr.t_release + self.FLICK_RETURN:
                    s = (t - tr.t_release) / self.FLICK_RETURN
                    a_rel = tr.angle(tr.t_release)
                    p_rel = rot(ax, a_rel) @ contact
                    # the finger keeps its follow-through briefly, then retracts
                    ang = a_rel + tr.sgn * math.radians(14) * math.sin(math.pi * min(1.0, s * 1.6) / 2) * (1 - s)
                    p_fol = rot(ax, ang) @ contact
                    out = p_fol + (p_fol - Vector((0, 0, p_fol.z))).normalized() * 0.006 * math.sin(math.pi * s)
                    targets[fn] = out.lerp(hov, smooth(s))
        return palm, targets


# ----------------------------------------------------------------------------- keyed poses
class Track:
    """Piecewise min-jerk interpolation between keyed 4x4 poses."""

    def __init__(self):
        self.keys: list[tuple[float, Matrix]] = []

    def key(self, t, m):
        self.keys.append((t, m.copy()))
        self.keys.sort(key=lambda k: k[0])

    def __call__(self, t, arc=0.0, arc_dir=Vector((0, 0, 1))):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1].copy()
        if t >= ks[-1][0]:
            return ks[-1][1].copy()
        for (ta, A), (tb, B) in zip(ks, ks[1:]):
            if ta <= t <= tb:
                s = smooth((t - ta) / (tb - ta)) if tb > ta else 1.0
                m = mat_lerp(A, B, s)
                return m
        return ks[-1][1].copy()


def smooth_noise(t, seed, freqs=(0.31, 0.53, 0.97, 1.7)):
    rng = random.Random(seed)
    v = 0.0
    for f in freqs:
        ph = rng.uniform(0, 2 * math.pi)
        v += math.sin(2 * math.pi * f * t + ph) / (1 + f)
    return v / 2.0


# ----------------------------------------------------------------------------- master
class Choreography:
    def __init__(self, seed=7):
        self.scramble, self.solution, self.seed = find_scramble_and_solution(seed)
        st = CubeState()
        st.apply_alg(self.scramble)
        self.R_init = [r.copy() for r in st.R]
        self.state0 = st

        # ---------------------------------------------------------------- timing
        T = {}
        T["eyes_on"] = 0.35
        T["reach0"] = 1.35
        T["pregrasp"] = 2.25
        T["grasp"] = 2.55
        T["lift0"] = 2.70
        T["lift1"] = 3.55
        T["insp_pos"] = 4.10
        # inspection: four x' style whole-cube rotations with regrips
        insp = []
        t = 4.35
        for k in range(4):
            r0, r1 = t, t + 0.52
            hold = 0.46 if k < 3 else 0.30
            g0 = r1 + hold
            gR = (g0, g0 + 0.28)
            gL = (g0 + 0.22, g0 + 0.50)
            insp.append(dict(rot=(r0, r1), regripR=gR, regripL=gL))
            t = gL[1] + 0.06
        T["insp"] = insp
        T["insp_end"] = t
        T["to_solve"] = (t, t + 0.55)
        T["think"] = (t + 0.55, t + 0.95)
        T["solve0"] = t + 0.95
        self.T = T
        self.solve = Solve(st, self.solution, T["solve0"])
        ts_end = self.solve.t_end
        T["solved"] = ts_end
        T["present"] = (ts_end + 0.45, ts_end + 1.75)
        T["down"] = (ts_end + 1.95, ts_end + 2.95)
        T["release"] = (ts_end + 2.95, ts_end + 3.25)
        T["retract"] = (ts_end + 3.25, ts_end + 4.25)
        T["look_cam"] = ts_end + 3.6
        T["end"] = ts_end + 5.3
        self.duration = T["end"]

        # ---------------------------------------------------------------- holding frame
        H = Track()
        h_table = Matrix.Translation(CUBE_REST_POS) @ CUBE_UPRIGHT
        tilt = lambda deg: Matrix.Rotation(math.radians(deg), 4, "X")
        h_hold = Matrix.Translation((0.0, -0.315, 0.965)) @ CUBE_UPRIGHT @ tilt(18)
        h_insp = Matrix.Translation((0.0, -0.285, 1.165)) @ CUBE_UPRIGHT @ tilt(32)
        h_solve = Matrix.Translation((0.0, -0.300, 0.995)) @ CUBE_UPRIGHT @ tilt(24)
        h_present = Matrix.Translation((0.0, -0.335, 1.04)) @ Matrix.Rotation(math.radians(24), 4, "Z") @ CUBE_UPRIGHT @ tilt(8)
        h_present2 = Matrix.Translation((0.0, -0.335, 1.04)) @ Matrix.Rotation(math.radians(-20), 4, "Z") @ CUBE_UPRIGHT @ tilt(8)
        h_table2 = Matrix.Translation(CUBE_REST_POS + Vector((0.01, -0.012, 0))) @ Matrix.Rotation(math.radians(4), 4, "Z") @ CUBE_UPRIGHT
        H.key(0.0, h_table)
        H.key(T["lift0"], h_table)
        lift_mid = Matrix.Translation((0.0, -0.36, 0.83)) @ CUBE_UPRIGHT @ tilt(6)
        H.key(T["lift0"] + 0.40, lift_mid)
        H.key(T["lift1"], h_hold)
        H.key(T["insp_pos"], h_insp)
        H.key(T["insp_end"], h_insp)
        H.key(T["to_solve"][1], h_solve)
        H.key(ts_end + 0.30, h_solve)
        H.key(T["present"][0] + 0.45, h_present)
        H.key(T["present"][1], h_present2)
        H.key(T["down"][0] + 0.55, Matrix.Translation(CUBE_REST_POS + Vector((0.01, -0.012, 0.035))) @ Matrix.Rotation(math.radians(4), 4, "Z") @ CUBE_UPRIGHT)
        H.key(T["down"][1], h_table2)
        self.H = H
        self.h_table2 = h_table2

        # ---------------------------------------------------------------- rest poses
        # hands resting on the table, palms down, fingers relaxed toward the cube
        yv = Vector((0.28, -1.0, -0.10))
        zv = Vector((-0.10, 0.0, 1.0))
        rest_R = hand_frame(yv.cross(zv), yv, (-0.215, -0.290, TABLE_TOP_Z + 0.036))
        self.rest = {"R": rest_R, "L": mirror_m(rest_R)}

        self._precompute_secondary()

    # --------------------------------------------------------------- cube body
    def Q(self, t):
        """Cube body rotation relative to H (inspection rotations about H x-axis)."""
        q = Matrix.Identity(4)
        for k, ins in enumerate(self.T["insp"]):
            r0, r1 = ins["rot"]
            if t >= r1:
                q = rot(0, P.QUARTER) @ q
            elif t > r0:
                q = rot(0, P.QUARTER * smooth((t - r0) / (r1 - r0))) @ q
        return q

    def H_at(self, t):
        h = self.H(t)
        # subtle handling micro-motion while held (servo hold, not perfectly still)
        if self.T["lift0"] < t < self.T["down"][1]:
            amp = 0.0006 if self.solve.t_start - 0.3 < t < self.solve.t_end else 0.0012
            w = min(1.0, (t - self.T["lift0"]) / 0.5, (self.T["down"][1] - t) / 0.5)
            n = Vector((smooth_noise(t, 11), smooth_noise(t, 12), smooth_noise(t, 13))) * amp * w
            rn = Matrix.Rotation(0.004 * w * smooth_noise(t, 14), 4, "Z") @ Matrix.Rotation(0.004 * w * smooth_noise(t, 15), 4, "X")
            h = Matrix.Translation(n) @ h @ rn
        if self.solve.t_start - 0.05 < t < self.solve.t_end + 0.6:
            h = h @ self.solve.recoil_matrix(t)
        return h

    def cube_world(self, t):
        return self.H_at(t) @ self.Q(t)

    def cubie_rotations(self, t):
        return self.solve.cubie_rotations(t, self.R_init)

    # --------------------------------------------------------------- hands
    def _held_hand(self, side, t):
        """Palm frame (H coords) + finger targets (H coords) while holding."""
        T = self.T
        mir = side == "L"
        gs = G_SOLVE_R if not mir else grip_L(G_SOLVE_R)
        gp = G_PICK_R if not mir else grip_L(G_PICK_R)

        def C(name):
            p, n = CONTACT[name]
            return mirror_v(p) if mir else p.copy()

        def V(dic, name):
            return mirror_v(dic[name]) if mir else dic[name].copy()

        # grip blend pick -> solve while lifting, solve -> pick while setting down
        if t < T["lift1"]:
            w = smooth((t - T["lift0"]) / (T["lift1"] - T["lift0"]))
        elif t > T["down"][0]:
            w = 1 - smooth((t - T["down"][0]) / (T["down"][1] - T["down"][0]))
        else:
            w = 1.0
        palm = mat_lerp(gp, gs, w)
        targets = {"thumb": C("thumb"), "middle": C("middle"), "index": V(HOVER, "index"), "ring": V(HOVER, "ring")}
        if T["lift1"] > t or t > T["down"][0]:
            for k in ("index", "ring"):
                targets[k] = targets[k].lerp(V(RETRACT, k), 1 - w)

        if self.solve.t_start - 0.1 <= t <= self.solve.t_end + 0.05:
            palm, targets = self.solve.hand_state(side, t)
        else:
            # inspection rotations / regrips (about H x-axis)
            for ins in T["insp"]:
                r0, r1 = ins["rot"]
                g0, g1 = ins["regripR"] if side == "R" else ins["regripL"]
                if r0 <= t < g0:
                    a = P.QUARTER * smooth((t - r0) / (r1 - r0)) if t < r1 else P.QUARTER
                    palm = rot(0, WRIST_PALM_RATIO * a) @ palm
                    for k in targets:
                        targets[k] = rot(0, a) @ targets[k]
                    break
                if g0 <= t < g1:
                    s = (t - g0) / (g1 - g0)
                    palm = rot(0, WRIST_PALM_RATIO * P.QUARTER * (1 - smooth(s))) @ palm
                    for k in targets:
                        targets[k] = regrip_path(0, targets[k], P.QUARTER, s, 0.009)
                    break
                if r1 <= t < g1:
                    break
        return palm, targets

    def hand(self, side, t):
        """World palm frame + per-finger command: ('target', world point) or ('pose', angles)."""
        T = self.T
        rest = self.rest[side]
        mir = side == "L"
        gp = G_PICK_R if not mir else grip_L(G_PICK_R)
        cmds = {}
        # pre-grasp: palm offset outward & up from the pick grip at the table pose
        out = Vector((0.035 * (-1 if mir else 1), 0.0, 0.0))
        pre_local = Matrix.Translation(out + Vector((0, -0.01, 0.03))) @ gp
        H_table = self.H(0.0)
        pre_world = H_table @ pre_local
        grasp_world = H_table @ gp

        def rest_cmds(w_open=0.0):
            c = {}
            for fn in FINGERS:
                c[fn] = ("pose", POSE_RELAX[fn])
            return c

        if t < T["reach0"]:
            return rest, ("rest", None)
        if t < T["pregrasp"]:
            s = smooth((t - T["reach0"]) / (T["pregrasp"] - T["reach0"]))
            m = mat_lerp(rest, pre_world, s)
            m.translation += Vector((0, 0, 0.05 * math.sin(math.pi * s)))
            return m, ("blend_rest_open", s)
        if t < T["grasp"]:
            s = smooth((t - T["pregrasp"]) / (T["grasp"] - T["pregrasp"]))
            m = mat_lerp(pre_world, grasp_world, s)
            return m, ("blend_open_grip", s)
        if t < T["release"][0]:
            Hw = self.H_at(t)
            palm, targets = self._held_hand(side, t)
            return Hw @ palm, ("grip", (Hw, targets, 1.0))
        # release & retract
        Hw = self.H_at(t)
        palm, targets = self._held_hand(side, T["release"][0])
        g_world = Hw @ palm
        pre_world2 = self.H(T["down"][1]) @ pre_local
        if t < T["release"][1]:
            s = smooth((t - T["release"][0]) / (T["release"][1] - T["release"][0]))
            m = mat_lerp(g_world, pre_world2, s)
            return m, ("grip_release", (Hw, targets, s))
        s = smooth((t - T["retract"][0]) / (T["retract"][1] - T["retract"][0]))
        m = mat_lerp(pre_world2, rest, s)
        m.translation += Vector((0, 0, 0.04 * math.sin(math.pi * s)))
        return m, ("blend_open_rest", s)

    # --------------------------------------------------------------- body
    def _precompute_secondary(self):
        T = self.T
        dt = 1 / 240
        n = int(self.duration / dt) + 2
        ts = [i * dt for i in range(n)]
        lean_raw, look_raw = [], []
        cam_target = CAM_END_POS.copy()
        rng = random.Random(5)
        sacc_t = 0.0
        sacc = Vector((0, 0, 0))
        for t in ts:
            # torso lean (radians, forward positive)
            if t < T["reach0"]:
                lean = 0.05
            elif t < T["lift1"]:
                lean = 0.21
            elif t < T["insp_end"]:
                lean = 0.03
            elif t < T["solved"]:
                lean = 0.09
            elif t < T["down"][0]:
                lean = 0.05
            elif t < T["retract"][0]:
                lean = 0.19
            else:
                lean = 0.04
            lean_raw.append(lean)
            c = self.cube_world(t).translation
            if T["insp_pos"] < t < T["insp_end"]:
                if t - sacc_t > rng.uniform(0.22, 0.38):
                    sacc_t = t
                    sacc = Vector((rng.uniform(-0.018, 0.018), rng.uniform(-0.01, 0.01), rng.uniform(-0.018, 0.018)))
                c = c + sacc
            elif t < T["reach0"] and t > 0.6:
                if t - sacc_t > rng.uniform(0.35, 0.6):
                    sacc_t = t
                    sacc = Vector((rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), 0))
                c = c + sacc
            if t > T["look_cam"]:
                w = smooth((t - T["look_cam"]) / 0.9)
                c = c.lerp(cam_target, w)
            look_raw.append(tuple(c))
        self.sec_ts = ts
        self.lean = P.Smoother(9.0).run(ts, lean_raw)
        self.look = P.Smoother(16.0).run(ts, look_raw)

    def _interp(self, arr, t):
        dt = self.sec_ts[1]
        x = t / dt
        i = max(0, min(len(arr) - 2, int(x)))
        a = x - i
        A, B = arr[i], arr[i + 1]
        if isinstance(A, tuple):
            return Vector(A).lerp(Vector(B), a)
        return A + (B - A) * a

    def lean_at(self, t):
        return self._interp(self.lean, t) + 0.006 * smooth_noise(t, 21)

    def look_at(self, t):
        return self._interp(self.look, t)

    def eye_intensity(self, t):
        T = self.T
        if t < T["eyes_on"]:
            return 0.0
        if t < T["eyes_on"] + 0.5:
            s = (t - T["eyes_on"]) / 0.5
            flick = 1.0 if (int(s * 9) % 3 != 1) else 0.25
            return 7.0 * smooth(s) * flick
        base = 6.0
        if T["insp_pos"] < t < T["insp_end"]:
            base = 8.0 + 1.5 * math.sin(2 * math.pi * 3.0 * t)
        t0, t1 = T["think"]
        if t0 < t < t1 + 0.15:
            s = (t - t0) / (t1 + 0.15 - t0)
            base = 6.0 + 10.0 * math.sin(math.pi * s) ** 2
        return base

"""Analytic IK: 2-bone arm with pole vector + coupled 3-link planar fingers."""
from __future__ import annotations

import math

from mathutils import Matrix, Quaternion, Vector

from .rig_def import FINGERS, FORE_LEN, UPPER_LEN, WRIST_ROLL_AT, finger_rest_frame, pad_point


def frame(x: Vector, y: Vector, origin: Vector) -> Matrix:
    """4x4 from orthonormal X, Y axes (Z = X x Y) and origin."""
    x = x.normalized()
    y = (y - x * y.dot(x)).normalized()
    z = x.cross(y)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = origin
    return m


def solve_arm(S: Vector, hand_world: Matrix, pole: Vector):
    """Return world matrices (upper, forearm, wrist_roll, hand) for a right-handed arm.

    Frames: Y along segment, X = elbow hinge axis.
    The hand is placed exactly at hand_world if reachable; otherwise the arm
    stretches toward it and the hand keeps its orientation.
    """
    W = hand_world.translation.copy()
    L1, L2 = UPPER_LEN, FORE_LEN
    d = W - S
    dist = d.length
    reach = (L1 + L2) * 0.9995
    unreachable = dist > reach
    dist_c = min(max(dist, abs(L1 - L2) + 1e-4), reach)
    k = d.normalized()
    cos_a = (L1 * L1 + dist_c * dist_c - L2 * L2) / (2 * L1 * dist_c)
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    p = pole - k * pole.dot(k)
    if p.length < 1e-6:
        p = Vector((0, 0, -1)) - k * k.z
    p.normalize()
    E = S + L1 * (math.cos(a) * k + math.sin(a) * p)
    Wc = S + k * dist_c
    up_dir = (E - S).normalized()
    fo_dir = (Wc - E).normalized()
    hinge = up_dir.cross(fo_dir)
    if hinge.length < 1e-6:
        hinge = up_dir.cross(p)
    hinge.normalize()
    upper = frame(hinge, up_dir, S)
    fore = frame(hinge, fo_dir, E)
    # forearm roll so the wrist cuff follows the hand's twist
    hrot = hand_world.to_3x3()
    rel = fore.to_3x3().transposed() @ hrot
    xh = rel.col[0]
    roll = math.atan2(-xh.z, xh.x)
    roll_m = Matrix.Rotation(roll, 4, "Y")
    wroll = fore @ Matrix.Translation((0, WRIST_ROLL_AT, 0)) @ roll_m
    hand = hand_world.copy()
    if unreachable:
        hand.translation = Wc
    return upper, fore, wroll, hand, unreachable


# ---------------------------------------------------------------------- fingers
def _chain_point(name, q1, q2):
    f = FINGERS[name]
    l1, l2, l3 = f["lens"]
    c = f["couple"]
    q3 = c * q2
    pp = pad_point(name)  # (0, along, -palmar) in distal frame
    along, depth = pp.y, -pp.z
    u = l1 * math.cos(q1) + l2 * math.cos(q1 + q2)
    v = l1 * math.sin(q1) + l2 * math.sin(q1 + q2)
    phi = q1 + q2 + q3
    u += along * math.cos(phi) - depth * math.sin(phi)
    v += along * math.sin(phi) + depth * math.cos(phi)
    return u, v


_REST = {}


def rest_frame(name):
    if name not in _REST:
        _REST[name] = finger_rest_frame(name)
    return _REST[name]


LIMITS = {
    "thumb": dict(psi=(-0.7, 0.9), q1=(-0.5, 1.2), q2=(-0.1, 1.4)),
    "default": dict(psi=(-0.30, 0.30), q1=(-0.35, 1.65), q2=(0.0, 1.75)),
}


def finger_ik(name, target_hand: Vector):
    """Target pad position in canonical right-hand space -> (psi, q1, q2)."""
    f = FINGERS[name]
    lim = LIMITS["thumb" if name == "thumb" else "default"]
    R = rest_frame(name)
    T = R.transposed() @ (target_hand - Vector(f["base"]))
    # abduction about local Z (dorsal), aligning the flexion plane with the target
    # the pad may sit in front of (u>0) or behind (u<0, curled) the base joint
    cands = [math.atan2(-T.x, T.y), math.atan2(T.x, -T.y)]
    best = None
    for c in cands:
        cc = max(lim["psi"][0], min(lim["psi"][1], c))
        cost = abs(cc - c) * 10 + abs(cc)
        if best is None or cost < best[0]:
            best = (cost, cc)
    psi = best[1]
    u = -T.x * math.sin(psi) + T.y * math.cos(psi)
    v = -T.z
    rho = math.hypot(u, v)
    lo, hi = lim["q2"]
    # distance from base is monotone decreasing in q2 -> bisection
    def D(q2):
        a, b = _chain_point(name, 0.0, q2)
        return math.hypot(a, b)

    if rho >= D(lo):
        q2 = lo
    elif rho <= D(hi):
        q2 = hi
    else:
        a_, b_ = lo, hi
        for _ in range(40):
            mid = 0.5 * (a_ + b_)
            if D(mid) > rho:
                a_ = mid
            else:
                b_ = mid
        q2 = 0.5 * (a_ + b_)
    pu, pv = _chain_point(name, 0.0, q2)
    q1 = math.atan2(v, u) - math.atan2(pv, pu)
    q1 = (q1 + math.pi) % (2 * math.pi) - math.pi
    q1 = max(lim["q1"][0], min(lim["q1"][1], q1))
    return psi, q1, q2


def finger_locals(name, psi, q1, q2):
    """Local rotation quaternions for the three finger joint empties (canonical right hand)."""
    f = FINGERS[name]
    R = rest_frame(name).to_4x4()
    m1 = R @ Matrix.Rotation(psi, 4, "Z") @ Matrix.Rotation(-q1, 4, "X")
    m2 = Matrix.Rotation(-q2, 4, "X")
    m3 = Matrix.Rotation(-f["couple"] * q2, 4, "X")
    return m1.to_quaternion(), m2.to_quaternion(), m3.to_quaternion()


def finger_pad_world(name, psi, q1, q2):
    """Forward kinematics: pad position in canonical hand space (for checks)."""
    f = FINGERS[name]
    q_1, q_2, q_3 = finger_locals(name, psi, q1, q2)
    m = Matrix.Translation(f["base"]) @ q_1.to_matrix().to_4x4()
    m = m @ Matrix.Translation((0, f["lens"][0], 0)) @ q_2.to_matrix().to_4x4()
    m = m @ Matrix.Translation((0, f["lens"][1], 0)) @ q_3.to_matrix().to_4x4()
    return m @ pad_point(name)

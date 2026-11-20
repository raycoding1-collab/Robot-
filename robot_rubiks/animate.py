"""Evaluate the choreography + IK at every (sub)frame and bake keyframes."""
from __future__ import annotations

import math
import time

import bpy
from mathutils import Matrix, Quaternion, Vector

from . import ik
from .choreography import FINGERS, POSE_CURL_PINKY, POSE_OPEN, POSE_RELAX, Choreography, mirror_m, smooth_noise
from .rig_def import (
    CHEST_OFF, EYE_OFFSET, FINGERS as FDEF, FPS, HEAD_OFF, MIRROR, NECK_OFF, PELVIS_POS,
    SHOULDER_OFF_R, SPINE_OFF, TABLE_TOP_Z,
)

S4 = MIRROR
S3 = Matrix.Diagonal((-1.0, 1.0, 1.0))
POLE_R = Vector((-0.55, 0.75, -1.0)).normalized()


class KeyBuffer:
    """Collects (frame, value) samples per (object, data_path, index); writes in bulk."""

    def __init__(self):
        self.data = {}
        self.last_q = {}

    def add_matrix(self, obj, frame, m: Matrix):
        loc, q, _ = m.decompose()
        prev = self.last_q.get(obj.name)
        if prev is not None and prev.dot(q) < 0:
            q = -q
        self.last_q[obj.name] = q
        for i in range(3):
            self.data.setdefault((obj, "location", i), []).append((frame, loc[i]))
        for i in range(4):
            self.data.setdefault((obj, "rotation_quaternion", i), []).append((frame, q[i]))

    def add_value(self, owner, path, frame, value, index=0):
        self.data.setdefault((owner, path, index), []).append((frame, value))

    def write(self):
        actions = {}
        for (owner, path, index), samples in self.data.items():
            if owner not in actions:
                owner.animation_data_create()
                act = bpy.data.actions.new(f"ACT_{owner.name}")
                owner.animation_data.action = act
                actions[owner] = act
            act = actions[owner]
            fc = act.fcurve_ensure_for_datablock(owner, path, index=index)
            fc.keyframe_points.clear()
            fc.keyframe_points.add(len(samples))
            flat = []
            for f, v in samples:
                flat += [f, v]
            fc.keyframe_points.foreach_set("co", flat)
            fc.keyframe_points.foreach_set("interpolation", [1] * len(samples))  # LINEAR
            fc.update()


def _rest_targets_canon(hand_world_c: Matrix):
    """Fingertips resting on the table, in canonical hand space (hand_world_c is canonical-space world)."""
    out = {}
    inv = hand_world_c.inverted()
    for fn in FINGERS:
        f = FDEF[fn]
        base = Vector(f["base"])
        if fn == "thumb":
            p = base + Vector((-0.03, 0.045, -0.03))
        else:
            reach = sum(f["lens"]) * 0.80
            p = base + Vector((0.0, reach, -0.02))
        pw = hand_world_c @ p
        pw.z = TABLE_TOP_Z + 0.0002 + (0.006 if fn == "thumb" else 0.0)
        out[fn] = inv @ pw
    return out


class Rig:
    def __init__(self, J, choreo: Choreography):
        self.J = J
        self.c = choreo
        self.unreachable = 0
        self.max_finger_err = {}

    # ---------------------------------------------------------------- torso/head
    def torso(self, t):
        lean = self.c.lean_at(t)
        breathe = 0.004 * math.sin(2 * math.pi * 0.23 * t)
        pel = Matrix.Translation(PELVIS_POS)
        spine_l = Matrix.Translation(SPINE_OFF) @ Matrix.Rotation(0.6 * lean + breathe, 4, "X") @ Matrix.Rotation(0.01 * smooth_noise(t, 31), 4, "Z")
        chest_l = Matrix.Translation(CHEST_OFF) @ Matrix.Rotation(0.4 * lean, 4, "X") @ Matrix.Rotation(0.008 * smooth_noise(t, 32), 4, "Y")
        spine = pel @ spine_l
        chest = spine @ chest_l
        return pel, spine_l, chest_l, spine, chest

    def head(self, t, chest):
        target = self.c.look_at(t)
        neck0 = chest @ Matrix.Translation(NECK_OFF)
        eye = neck0 @ Matrix.Translation(HEAD_OFF) @ EYE_OFFSET
        d = chest.to_3x3().inverted() @ (target - eye)
        yaw = math.atan2(d.x, -d.y)
        pitch = math.atan2(-d.z, math.hypot(d.x, d.y))
        yaw += 0.012 * smooth_noise(t, 41)
        pitch += 0.010 * smooth_noise(t, 42)
        roll = 0.025 * smooth_noise(t, 43, (0.13, 0.29))
        neck_l = Matrix.Translation(NECK_OFF) @ Matrix.Rotation(0.35 * yaw, 4, "Z") @ Matrix.Rotation(0.30 * pitch, 4, "X")
        head_l = Matrix.Translation(HEAD_OFF) @ Matrix.Rotation(0.65 * yaw, 4, "Z") @ Matrix.Rotation(0.70 * pitch, 4, "X") @ Matrix.Rotation(roll, 4, "Y")
        return neck_l, head_l

    # ---------------------------------------------------------------- arms
    def arm(self, side, chest, hand_world):
        sh = SHOULDER_OFF_R.copy()
        if side == "L":
            sh.x = -sh.x
        S = chest @ sh
        if side == "R":
            up, fo, wr, hd, bad = ik.solve_arm(S, hand_world, POLE_R)
        else:
            up, fo, wr, hd, bad = ik.solve_arm(S3 @ S, mirror_m(hand_world), POLE_R)
            up, fo, wr, hd = (mirror_m(x) for x in (up, fo, wr, hd))
        if bad:
            self.unreachable += 1
        return up, fo, wr, hd

    # ---------------------------------------------------------------- fingers
    def finger_angles(self, side, hand_w, cmd):
        """Return dict finger -> (psi, q1, q2) in canonical space."""
        kind, data = cmd
        hand_c = hand_w if side == "R" else mirror_m(hand_w)  # canonical-space world

        def rest_angles():
            tg = _rest_targets_canon(hand_c)
            return {fn: ik.finger_ik(fn, tg[fn]) for fn in FINGERS}

        def grip_angles(Hw, targets):
            inv = hand_w.inverted()
            out = {}
            for fn in FINGERS:
                if fn == "pinky":
                    out[fn] = POSE_CURL_PINKY
                    continue
                pw = Hw @ targets[fn]
                q = inv @ pw
                if side == "L":
                    q = S3 @ q
                out[fn] = ik.finger_ik(fn, q)
                # bookkeeping: residual error
                pad = ik.finger_pad_world(fn, *out[fn])
                err = (pad - q).length
                key = (side, fn)
                self.max_finger_err[key] = max(self.max_finger_err.get(key, 0.0), err)
            return out

        def lerp_angles(A, B, s):
            return {fn: tuple(a + (b - a) * s for a, b in zip(A[fn], B[fn])) for fn in FINGERS}

        if kind == "rest":
            return rest_angles()
        if kind == "blend_rest_open":
            return lerp_angles(rest_angles(), POSE_OPEN, data)
        if kind == "blend_open_grip":
            t = self._t
            palm, targets = self.c._held_hand(side, t)
            g = grip_angles(self.c.H_at(t), targets)
            return lerp_angles(POSE_OPEN, g, data)
        if kind == "grip":
            Hw, targets, _ = data
            return grip_angles(Hw, targets)
        if kind == "grip_release":
            Hw, targets, s = data
            return lerp_angles(grip_angles(Hw, targets), POSE_OPEN, s)
        if kind == "blend_open_rest":
            return lerp_angles(POSE_OPEN, rest_angles(), data)
        raise ValueError(kind)

    # ---------------------------------------------------------------- full pose
    def evaluate(self, t):
        self._t = t
        J = self.J
        out = {}
        pel, spine_l, chest_l, spine, chest = self.torso(t)
        out["spine"] = spine_l
        out["chest"] = chest_l
        neck_l, head_l = self.head(t, chest)
        out["neck"] = neck_l
        out["head"] = head_l
        for side in ("R", "L"):
            hand_w, cmd = self.c.hand(side, t)
            up, fo, wr, hd = self.arm(side, chest, hand_w)
            out[f"uparm_{side}"] = chest.inverted() @ up
            out[f"forearm_{side}"] = up.inverted() @ fo
            out[f"wroll_{side}"] = fo.inverted() @ wr
            out[f"hand_{side}"] = wr.inverted() @ hd
            angles = self.finger_angles(side, hd, cmd)
            for fn in FINGERS:
                q1, q2, q3 = ik.finger_locals(fn, *angles[fn])
                base = Vector(FDEF[fn]["base"])
                lens = FDEF[fn]["lens"]
                m1 = Matrix.Translation(base) @ q1.to_matrix().to_4x4()
                m2 = Matrix.Translation((0, lens[0], 0)) @ q2.to_matrix().to_4x4()
                m3 = Matrix.Translation((0, lens[1], 0)) @ q3.to_matrix().to_4x4()
                if side == "L":
                    m1, m2, m3 = (S4 @ m @ S4 for m in (m1, m2, m3))
                out[f"{fn}1_{side}"] = m1
                out[f"{fn}2_{side}"] = m2
                out[f"{fn}3_{side}"] = m3
        return out


def sample_times(choreo: Choreography, sub_fast=2):
    """Frame numbers to key: every frame, plus sub-frames during fast motion."""
    n = int(math.ceil(choreo.duration * FPS)) + 1
    frames = []
    s0 = choreo.solve.t_start * FPS - 3
    s1 = choreo.solve.t_end * FPS + 3
    for f in range(1, n + 1):
        frames.append(float(f))
        if s0 <= f <= s1:
            for k in range(1, sub_fast):
                frames.append(f + k / sub_fast)
    return frames


def bake(choreo: Choreography, J, cube_root, cubies, eye_mat=None, sub_fast=4, verbose=True):
    rig = Rig(J, choreo)
    kb = KeyBuffer()
    frames = sample_times(choreo, sub_fast)
    t_start = time.time()
    for fr in frames:
        t = (fr - 1) / FPS
        pose = rig.evaluate(t)
        for name, m in pose.items():
            kb.add_matrix(J[name], fr, m)
        kb.add_matrix(cube_root, fr, choreo.cube_world(t))
        rots = choreo.cubie_rotations(t)
        for obj, m in zip(cubies, rots):
            kb.add_matrix(obj, fr, m)
    if eye_mat is not None:
        node = eye_mat.node_tree.nodes.get("Principled BSDF")
        sock = node.inputs["Emission Strength"]
        for f in range(1, int(choreo.duration * FPS) + 2):
            t = (f - 1) / FPS
            kb.add_value(eye_mat.node_tree, sock.path_from_id("default_value"), f, choreo.eye_intensity(t))
    kb.write()
    if verbose:
        print(f"baked {len(frames)} samples in {time.time() - t_start:.1f}s; unreachable arm samples: {rig.unreachable}")
        worst = sorted(rig.max_finger_err.items(), key=lambda kv: -kv[1])
        print("max fingertip residual (mm):", {f"{k[0]}:{k[1]}": round(v * 1000, 1) for k, v in worst})
    return rig, frames

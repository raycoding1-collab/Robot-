"""Procedural hard-surface humanoid robot.

Every rigid part is a mesh parented to a joint Empty; the animation keys the
Empties (exact rigid-body motion -> exact motion blur, no skinning artefacts).
"""
from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import geo
from . import materials as M
from .rig_def import (
    CHEST_OFF, FINGERS, FINGER_ORDER, FORE_LEN, HEAD_OFF, MIRROR, NECK_OFF, PELVIS_POS,
    SHOULDER_OFF_R, SPINE_OFF, UPPER_LEN, WRIST_ROLL_AT, finger_rest_frame,
)


class Mats:
    def __init__(self):
        self.white = M.ceramic_white()
        self.satin = M.satin_white()
        self.graphite = M.graphite_metal()
        self.dark = M.dark_polymer()
        self.chrome = M.chrome()
        self.rubber = M.rubber()
        self.visor = M.visor_glass()
        self.face = M.face_inner()
        self.lens = M.lens_glass()
        self.copper = M.copper()
        self.eye = M.emitter("Robot_EyeGlow", (0.35, 0.80, 1.0), 6.0)
        self.accent = M.emitter("Robot_Accent", (0.30, 0.75, 1.0), 3.0)
        self.status = M.emitter("Robot_Status", (1.0, 0.55, 0.15), 2.0)


def _empty(name, parent, loc=(0, 0, 0), coll=None, size=0.03):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = size
    e.empty_display_type = "ARROWS"
    geo.link(e, coll)
    if parent:
        e.parent = parent
    e.location = loc
    e.rotation_mode = "QUATERNION"
    return e


class Builder:
    def __init__(self, coll, mats: Mats):
        self.coll = coll
        self.m = mats
        self.parts = []

    def add(self, name, mesh, mat, parent, mirror=False, smooth=True, sharp=None, xform=None):
        if mirror:
            mesh.transform(MIRROR)
            mesh.flip_normals()
        if xform is not None:
            mesh.transform(xform)
        obj = geo.new_object(name, mesh, mat, parent=parent, collection=self.coll, smooth=smooth, sharp_angle=sharp)
        self.parts.append(obj)
        return obj

    # ------------------------------------------------------------------ helpers
    def se(self, name, parent, center, r, e=(1, 1), mat=None, nu=48, nv=24, shape=None, rot=None, mirror=False, sharp=None):
        m4 = Matrix.Translation(center)
        if rot is not None:
            m4 = m4 @ rot
        me = geo.se_mesh(name, r[0], r[1], r[2], e[0], e[1], nu, nv, shape, m4)
        return self.add(name, me, mat or self.m.white, parent, mirror, sharp=sharp)

    def cyl(self, name, parent, center, r, depth, axis="Z", mat=None, bevel=0.0, segs=40, mirror=False):
        me = geo.cylinder_mesh(name, r, depth, segs=segs, axis=axis, bevel=bevel, center=center)
        return self.add(name, me, mat or self.m.graphite, parent, mirror, sharp=35)

    def torus(self, name, parent, center, R, r, axis="Z", mat=None, mirror=False, segs=56):
        me = geo.torus_mesh(name, R, r, segs=segs, rsegs=12, axis=axis, center=center)
        return self.add(name, me, mat or self.m.graphite, parent, mirror)

    def panel(self, name, parent, verts, faces, cuts, mat, thickness=0.0025, mirror=False, bevel=0.0007):
        me = geo.panel_mesh(name, verts, faces, cuts)
        obj = self.add(name, me, mat, parent, mirror)
        geo.add_solidify(obj, thickness, offset=-1.0)
        if bevel:
            geo.add_bevel(obj, bevel, 2, 45)
        return obj


# =============================================================================
def build_robot(coll, mats: Mats | None = None):
    mats = mats or Mats()
    B = Builder(coll, mats)
    J = {}

    root = _empty("RIG_root", None, (0, 0, 0), coll, 0.2)
    J["root"] = root
    J["pelvis"] = _empty("J_pelvis", root, PELVIS_POS, coll, 0.15)
    J["spine"] = _empty("J_spine", J["pelvis"], SPINE_OFF, coll, 0.12)
    J["chest"] = _empty("J_chest", J["spine"], CHEST_OFF, coll, 0.12)
    J["neck"] = _empty("J_neck", J["chest"], NECK_OFF, coll, 0.08)
    J["head"] = _empty("J_head", J["neck"], HEAD_OFF, coll, 0.08)

    for side in ("R", "L"):
        sx = 1 if side == "R" else -1
        sh = SHOULDER_OFF_R.copy()
        sh.x *= sx
        J[f"uparm_{side}"] = _empty(f"J_uparm_{side}", J["chest"], sh, coll, 0.06)
        J[f"forearm_{side}"] = _empty(f"J_forearm_{side}", J[f"uparm_{side}"], (0, UPPER_LEN, 0), coll, 0.05)
        J[f"wroll_{side}"] = _empty(f"J_wroll_{side}", J[f"forearm_{side}"], (0, WRIST_ROLL_AT, 0), coll, 0.04)
        J[f"hand_{side}"] = _empty(f"J_hand_{side}", J[f"wroll_{side}"], (0, FORE_LEN - WRIST_ROLL_AT, 0), coll, 0.04)
        for fn in FINGER_ORDER:
            f = FINGERS[fn]
            base = Vector(f["base"])
            if side == "L":
                base.x = -base.x
            p = _empty(f"J_{fn}1_{side}", J[f"hand_{side}"], base, coll, 0.012)
            J[f"{fn}1_{side}"] = p
            p2 = _empty(f"J_{fn}2_{side}", p, (0, f["lens"][0], 0), coll, 0.01)
            J[f"{fn}2_{side}"] = p2
            p3 = _empty(f"J_{fn}3_{side}", p2, (0, f["lens"][1], 0), coll, 0.008)
            J[f"{fn}3_{side}"] = p3

    _build_torso(B, J)
    _build_head(B, J)
    _build_legs(B, J)
    for side in ("R", "L"):
        _build_arm(B, J, side)
        _build_hand(B, J, side)
    return J, B.parts


# =============================================================================
def _build_torso(B: Builder, J):
    m = B.m
    pel = J["pelvis"]
    # pelvis shell
    def pel_shape(x, y, z, t):
        s = 0.92 + 0.08 * (t + 1) / 2
        return x * s, y, z

    B.se("pelvis_shell", pel, (0, 0.0, -0.005), (0.150, 0.108, 0.070), (0.5, 0.65), m.white, shape=pel_shape)
    B.se("pelvis_belt", pel, (0, 0.0, 0.058), (0.128, 0.092, 0.018), (0.3, 0.7), m.graphite)
    for sx in (-1, 1):
        B.cyl(f"hip_joint_{sx}", pel, (sx * 0.098, -0.01, -0.035), 0.056, 0.068, axis="X", mat=m.graphite, bevel=0.004)
        B.cyl(f"hip_cap_{sx}", pel, (sx * 0.134, -0.01, -0.035), 0.036, 0.008, axis="X", mat=m.chrome, bevel=0.0015)

    sp = J["spine"]
    # waist: dark ribbed flex core behind articulated white abdominal plates
    B.se("abdomen_core", sp, (0, 0.016, 0.082), (0.078, 0.062, 0.118), (0.6, 0.8), m.graphite, nu=56, nv=28)
    for i, z in enumerate((-0.004, 0.026, 0.056, 0.086, 0.116, 0.146, 0.176)):
        B.se(f"abdomen_rib_{i}", sp, (0, 0.016, z), (0.081, 0.065, 0.0028), (0.3, 0.8), m.dark, nu=56, nv=8)
    av, af = geo.superellipsoid(0.090, 0.074, 0.046, 0.5, 0.66, 64, 28)

    def ab_shape(x, y, z, t):
        return x * (0.92 + 0.08 * (t + 1) / 2), y, z

    av = [ab_shape(x, y, z, z / 0.046) for x, y, z in av]
    vv = [(x, y + 0.010, z + 0.085) for x, y, z in av]
    B.panel("ab_plate", sp, vv, af, [((0, -0.018, 0), (0, -1, 0))], m.white, 0.003)
    bv, bf = geo.superellipsoid(0.086, 0.074, 0.075, 0.5, 0.7, 56, 28)
    bv = [(x, y + 0.014, z + 0.085) for x, y, z in bv]
    B.panel("back_plate_low", sp, bv, bf, [((0, 0.048, 0), (0, 1, 0))], m.satin, 0.003)
    for sx in (-1, 1):  # side cable runs
        geo.curve_tube(f"abd_cable_{sx}", [(sx * 0.072, 0.05, -0.03), (sx * 0.09, 0.06, 0.08), (sx * 0.075, 0.06, 0.19)], 0.0055, m.rubber, sp, B.coll)

    ch = J["chest"]

    def chest_shape(x, y, z, t):
        s = (t + 1) / 2
        sx = 0.70 + 0.30 * min(1.0, s * 1.3)
        sy = 0.86 + 0.14 * math.sin(math.pi * min(1.0, s * 1.1))
        # pec bulge forward on the upper half
        yb = -0.012 * max(0.0, math.sin(math.pi * s)) if y < 0 else 0.0
        return x * sx, y * sy + yb, z

    C = (0, 0.016, 0.192)
    R = (0.162, 0.106, 0.158)
    v, f = geo.superellipsoid(*R, 0.44, 0.60, 80, 44, chest_shape)
    v = [(a + C[0], b + C[1], c + C[2]) for a, b, c in v]
    # dark inner core seen through the panel gaps
    vi = [(a * 0.965, (b - C[1]) * 0.955 + C[1], (c - C[2]) * 0.97 + C[2]) for a, b, c in v]
    B.add("chest_core", geo.mesh_from_data("chest_core", vi, f), m.dark, ch)
    g = 0.0016
    zsplit = 0.118
    # front-left/right upper plates, lower rib plates, back plate
    B.panel("chest_plate_R", ch, v, f, [((-g, 0, 0), (-1, 0, 0)), ((0, 0, zsplit + g), (0, 0, 1)), ((0, 0.01, 0), (0, -1, 0))], m.white, 0.003)
    B.panel("chest_plate_L", ch, v, f, [((g, 0, 0), (1, 0, 0)), ((0, 0, zsplit + g), (0, 0, 1)), ((0, 0.01, 0), (0, -1, 0))], m.white, 0.003)
    B.panel("chest_rib_R", ch, v, f, [((-g, 0, 0), (-1, 0, 0)), ((0, 0, zsplit - g), (0, 0, -1)), ((0, 0.01, 0), (0, -1, 0))], m.white, 0.003)
    B.panel("chest_rib_L", ch, v, f, [((g, 0, 0), (1, 0, 0)), ((0, 0, zsplit - g), (0, 0, -1)), ((0, 0.01, 0), (0, -1, 0))], m.white, 0.003)
    B.panel("chest_back", ch, v, f, [((0, 0.01 + g, 0), (0, 1, 0))], m.satin, 0.003)
    # sternum sensor + status light
    B.se("sternum_window", ch, (0, -0.103, 0.215), (0.014, 0.006, 0.022), (0.4, 0.4), m.lens, nu=32, nv=12)
    B.se("status_led", ch, (0, -0.1055, 0.193), (0.0035, 0.002, 0.0012), (0.5, 0.5), m.status, nu=16, nv=8)
    # collar + neck socket
    B.torus("collar", ch, (0, 0.006, 0.322), 0.050, 0.011, mat=m.graphite)
    B.cyl("neck_socket", ch, (0, 0.006, 0.318), 0.044, 0.022, mat=m.dark, bevel=0.003)
    # shoulder sockets and caps
    for side, sx in (("R", 1), ("L", -1)):
        p = Vector(SHOULDER_OFF_R)
        p.x *= sx
        B.torus(f"shoulder_socket_{side}", ch, (p.x + sx * 0.012, p.y, p.z), 0.058, 0.012, axis="X", mat=m.graphite)
        cv, cf = geo.superellipsoid(0.060, 0.068, 0.050, 0.6, 0.7, 56, 28)
        cv = [(a + p.x - sx * 0.006, b + p.y, c + p.z + 0.022) for a, b, c in cv]
        B.panel(f"shoulder_cap_{side}", ch, cv, cf, [((0, 0, p.z + 0.004), (0, 0, 1))], m.white, 0.0035)
    # back power module with heat-sink fins
    B.se("back_module", ch, (0, 0.135, 0.19), (0.088, 0.035, 0.105), (0.3, 0.35), m.graphite, nu=48, nv=24)
    for i in range(9):
        x = -0.064 + i * 0.016
        geo_m = geo.rounded_box_mesh(f"fin_{i}", 0.004, 0.022, 0.17, 0.0012, 2, (x, 0.172, 0.19))
        B.add(f"fin_{i}", geo_m, m.graphite, ch, sharp=35)
    B.se("back_light", ch, (0, 0.171, 0.29), (0.05, 0.003, 0.0025), (0.4, 0.4), m.accent, nu=24, nv=8)
    # neck cables (static in chest frame, reaching up behind the neck)
    for sx in (-1, 1):
        geo.curve_tube(f"neck_cable_{sx}", [(sx * 0.03, 0.05, 0.30), (sx * 0.034, 0.055, 0.345), (sx * 0.026, 0.035, 0.40)], 0.0045, m.rubber, ch, B.coll)


def _build_head(B: Builder, J):
    m = B.m
    n_before = len(B.parts)
    nk = J["neck"]
    B.cyl("neck_column", nk, (0, 0, 0.042), 0.025, 0.09, mat=m.graphite, bevel=0.002)
    for i, z in enumerate((0.018, 0.042, 0.066)):
        B.torus(f"neck_ring_{i}", nk, (0, 0, z), 0.029, 0.0045, mat=m.dark)
    for sx in (-1, 1):
        B.cyl(f"neck_piston_{sx}", nk, (sx * 0.034, -0.022, 0.03), 0.0075, 0.05, mat=m.graphite, bevel=0.001)
        B.cyl(f"neck_rod_{sx}", nk, (sx * 0.034, -0.022, 0.07), 0.0035, 0.04, mat=m.chrome)

    hd = J["head"]
    C = (0.0, 0.012, 0.105)

    def skull(x, y, z, t):
        s = (t + 1) / 2
        k = 0.80 + 0.20 * min(1.0, s * 1.6)  # narrower jaw
        yb = y * (0.92 + 0.08 * s) if y < 0 else y
        return x * k, yb, z

    v, f = geo.superellipsoid(0.090, 0.106, 0.118, 0.82, 0.86, 96, 56, skull)
    v = [(a + C[0], b + C[1], c + C[2]) for a, b, c in v]
    # helmet: back shell + forehead + chin panels leave a band-shaped visor opening
    Y_CUT, Z_LO, Z_HI = -0.050, 0.044, 0.170
    B.panel("helmet_back", hd, v, f, [((0, Y_CUT + 0.0006, 0), (0, 1, 0))], m.white, 0.003, bevel=0.0008)
    B.panel("helmet_brow", hd, v, f, [((0, Y_CUT - 0.0006, 0), (0, -1, 0)), ((0, 0, Z_HI), (0, 0, 1))], m.white, 0.003, bevel=0.0008)
    B.panel("helmet_chin", hd, v, f, [((0, Y_CUT - 0.0006, 0), (0, -1, 0)), ((0, 0, Z_LO), (0, 0, -1))], m.white, 0.003, bevel=0.0008)
    # recessed dark face (smooth, glossy) behind the visor band
    fv = [((a - C[0]) * 0.945 + C[0], (b - C[1]) * 0.945 + C[1], (c - C[2]) * 0.97 + C[2]) for a, b, c in v]
    B.add("face_core", geo.mesh_from_data("face_core", fv, f), m.face, hd)
    # visor glass insert (real thickness, sits flush inside the band)
    gv = [((a - C[0]) * 0.995 + C[0], (b - C[1]) * 0.995 + C[1], c) for a, b, c in v]
    B.panel("visor", hd, gv, f, [((0, Y_CUT - 0.0004, 0), (0, -1, 0)), ((0, 0, Z_LO + 0.0025), (0, 0, 1)), ((0, 0, Z_HI - 0.0025), (0, 0, -1))], m.visor, 0.0018, bevel=0.0004)
    # eyes: thin camera modules between the dark face and the glass
    for sx in (-1, 1):
        ex, ez = sx * 0.031, 0.110
        rot = Matrix.Rotation(sx * math.radians(16), 4, "Z")
        B.add(f"eye_barrel_{sx}", geo.cylinder_mesh("eb", 0.0105, 0.003, 48, "Y", 0.0005, 2, (0, 0, 0)), m.graphite, hd, xform=Matrix.Translation((ex, -0.0798, ez)) @ rot, sharp=35)
        B.add(f"eye_lens_{sx}", geo.se_mesh("el", 0.0078, 0.0008, 0.0078, 1, 1, 40, 16), m.lens, hd, xform=Matrix.Translation((ex, -0.0812, ez)) @ rot)
        B.add(f"eye_iris_{sx}", geo.torus_mesh("ei", 0.0058, 0.0005, 48, 8, "Y"), m.eye, hd, xform=Matrix.Translation((ex, -0.0818, ez)) @ rot)
        B.add(f"eye_pupil_{sx}", geo.se_mesh("ep", 0.0014, 0.0004, 0.0014, 1, 1, 16, 8), m.eye, hd, xform=Matrix.Translation((ex, -0.0818, ez)) @ rot)
    for i in range(5):
        x = -0.012 + i * 0.006
        B.se(f"sensor_dot_{i}", hd, (x, -0.0805, 0.072), (0.0010, 0.0006, 0.0010), (1, 1), m.lens, nu=12, nv=6)
    # ear pods
    for sx in (-1, 1):
        B.cyl(f"ear_pod_{sx}", hd, (sx * 0.084, 0.014, 0.108), 0.030, 0.020, axis="X", mat=m.graphite, bevel=0.003)
        B.cyl(f"ear_disc_{sx}", hd, (sx * 0.0945, 0.014, 0.108), 0.020, 0.004, axis="X", mat=m.satin, bevel=0.0012)
        B.torus(f"ear_ring_{sx}", hd, (sx * 0.0952, 0.014, 0.108), 0.0245, 0.0011, axis="X", mat=m.accent)
    # crown seam (dark inset line across the top of the skull)
    B.cyl("head_base", hd, (0, 0.005, 0.012), 0.034, 0.024, mat=m.dark, bevel=0.003)
    # slightly smaller, more elegant head
    sc = Matrix.Diagonal((0.93, 0.93, 0.95, 1.0))
    for o in B.parts[n_before:]:
        if o.parent == hd and o.type == "MESH":
            o.data.transform(sc)


def _build_legs(B: Builder, J):
    m = B.m
    pel = J["pelvis"]
    for sx in (-1, 1):
        hip = Vector((sx * 0.098, -0.01, -0.035))
        knee = Vector((sx * 0.105, -0.405, -0.06))
        ankle = Vector((sx * 0.11, -0.44, -0.50))
        for name, a, b, r0, r1 in (("thigh", hip, knee, 0.072, 0.056), ("shin", knee, ankle, 0.054, 0.040)):
            d = b - a
            L = d.length
            rot = d.to_track_quat("Y", "Z").to_matrix().to_4x4()
            me = geo.limb_shell(f"{name}_{sx}", L * 0.84, r0, r1, flat=0.95, start=L * 0.08, bulge=0.06)
            me.transform(Matrix.Translation(a) @ rot)
            B.add(f"{name}_{sx}", me, m.white, pel)
        B.cyl(f"knee_{sx}", pel, tuple(knee), 0.047, 0.088, axis="X", mat=m.graphite, bevel=0.004)
        B.cyl(f"knee_cap_{sx}", pel, (knee.x + sx * 0.045, knee.y, knee.z), 0.030, 0.006, axis="X", mat=m.chrome, bevel=0.001)
        me = geo.rounded_box_mesh(f"foot_{sx}", 0.085, 0.23, 0.06, 0.02, 4, (ankle.x, ankle.y - 0.06, ankle.z - 0.06))
        B.add(f"foot_{sx}", me, m.graphite, pel, sharp=35)
        B.se(f"ankle_{sx}", pel, tuple(ankle), (0.035, 0.035, 0.035), (1, 1), m.graphite)


def _build_arm(B: Builder, J, side):
    m = B.m
    mir = side == "L"
    ua = J[f"uparm_{side}"]
    B.se(f"shoulder_ball_{side}", ua, (0, 0, 0), (0.051, 0.051, 0.051), (1, 1), m.graphite, nu=40, nv=20, mirror=mir)
    me = geo.limb_shell(f"upperarm_shell_{side}", 0.205, 0.053, 0.046, flat=0.96, start=0.048, bulge=0.05)
    B.add(f"upperarm_shell_{side}", me, m.white, ua, mir)
    B.torus(f"upperarm_band_{side}", ua, (0, 0.205, 0), 0.0445, 0.0045, axis="Y", mat=m.graphite, mirror=mir)
    B.cyl(f"biceps_actuator_{side}", ua, (0, 0.215, -0.030), 0.016, 0.07, axis="Y", mat=m.graphite, bevel=0.002, mirror=mir)
    B.cyl(f"biceps_rod_{side}", ua, (0, 0.255, -0.030), 0.006, 0.03, axis="Y", mat=m.chrome, mirror=mir)
    # elbow yoke
    for sx in (-1, 1):
        B.cyl(f"elbow_yoke_{side}_{sx}", ua, (sx * 0.041, UPPER_LEN, 0), 0.040, 0.012, axis="X", mat=m.graphite, bevel=0.002, mirror=mir)
        B.cyl(f"elbow_pin_{side}_{sx}", ua, (sx * 0.048, UPPER_LEN, 0), 0.018, 0.006, axis="X", mat=m.chrome, bevel=0.001, mirror=mir)
        me = geo.rounded_box_mesh(f"yoke_arm_{side}_{sx}", 0.011, 0.06, 0.05, 0.004, 3, (sx * 0.041, UPPER_LEN - 0.035, 0))
        B.add(f"yoke_arm_{side}_{sx}", me, m.graphite, ua, mir, sharp=35)

    fa = J[f"forearm_{side}"]
    B.cyl(f"elbow_hub_{side}", fa, (0, 0, 0), 0.035, 0.068, axis="X", mat=m.dark, bevel=0.003, mirror=mir)
    me = geo.limb_shell(f"forearm_shell_{side}", 0.13, 0.047, 0.043, flat=0.86, start=0.028, bulge=0.07)
    B.add(f"forearm_shell_{side}", me, m.white, fa, mir)
    B.cyl(f"wrist_roll_ring_{side}", fa, (0, WRIST_ROLL_AT - 0.002, 0), 0.040, 0.010, axis="Y", mat=m.graphite, bevel=0.002, mirror=mir)

    wr = J[f"wroll_{side}"]
    me = geo.limb_shell(f"wrist_cuff_{side}", 0.082, 0.040, 0.033, flat=0.80, start=0.006, bulge=0.03)
    B.add(f"wrist_cuff_{side}", me, m.satin, wr, mir)
    # dorsal light strip on the cuff
    me = geo.rounded_box_mesh(f"cuff_light_{side}", 0.0045, 0.036, 0.0024, 0.0009, 2, (0, 0.045, 0.0296))
    B.add(f"cuff_light_{side}", me, m.accent, wr, mir)
    B.se(f"wrist_ball_{side}", J[f"hand_{side}"], (0, -0.004, 0), (0.027, 0.027, 0.024), (1, 1), m.graphite, nu=32, nv=16, mirror=mir)


def _build_hand(B: Builder, J, side):
    m = B.m
    mir = side == "L"
    hd = J[f"hand_{side}"]

    def palm_shape(x, y, z, t):
        # widen towards the knuckles, slight arch
        s = min(1.0, max(0.0, (y - 0.004) / 0.094))  # 0 at wrist end, 1 at knuckles
        k = 0.82 + 0.18 * s
        return x * k, y, z + 0.006 * (1 - (x / 0.045) ** 2)

    # palm modelled along +Y: superellipsoid with long axis Z rotated to Y
    rot = geo.AXIS_Z_TO_Y
    v, f = geo.superellipsoid(0.043, 0.0155, 0.047, 0.42, 0.40, 56, 30)  # x width, y thickness(->z), z length(->y)
    v = geo.transform_verts(v, Matrix.Translation((0.001, 0.051, -0.001)) @ rot)
    v = [palm_shape(*p, 0) for p in v]
    # dorsal shell (white) + palmar pad (rubber) split
    B.panel(f"palm_dorsal_{side}", hd, v, f, [((0, 0, -0.0015), (0, 0, 1))], m.white, 0.003, mirror=mir)
    B.panel(f"palm_pad_{side}", hd, v, f, [((0, 0, -0.0005), (0, 0, -1))], m.rubber, 0.003, mirror=mir)
    vi = [(a * 0.96, b, (c + 0.001) * 0.9) for a, b, c in v]
    B.add(f"palm_core_{side}", geo.mesh_from_data("pc", vi, f), m.dark, hd, mir)
    B.cyl(f"carpal_{side}", hd, (0, 0.006, -0.001), 0.021, 0.02, axis="Y", mat=m.graphite, bevel=0.002, mirror=mir)
    # thenar housing at the thumb base
    tb = Vector(FINGERS["thumb"]["base"])
    B.se(f"thumb_socket_{side}", hd, tuple(tb), (0.0135, 0.0135, 0.0135), (1, 1), m.graphite, nu=28, nv=14, mirror=mir)

    for fn in FINGER_ORDER:
        f = FINGERS[fn]
        rest = finger_rest_frame(fn).to_4x4()
        w, t = f["width"], f["thick"]
        for k in range(3):
            jn = J[f"{fn}{k + 1}_{side}"]
            L = f["lens"][k]
            taper = (1.0, 0.93, 0.86)[k]
            ww, tt = w * taper, t * taper
            gap = 0.0042
            # NOTE: joint empties carry the finger rest frame in their animated rotation,
            # so geometry is modelled in the canonical joint frame (Y along segment).
            ln = L - 2 * gap if k < 2 else L - gap - 0.001
            cy = gap + ln / 2
            def ph_shape(x, y, z, tt_=0, ln_=ln, k_=k):
                s = (y / (ln_ / 2) + 1) / 2
                if k_ == 2:
                    kk = 1.0 - 0.18 * s ** 2
                    return x * kk, y, z * (1.0 - 0.12 * s ** 2) + 0.0015 * s
                return x, y, z
            v, fc = geo.superellipsoid(ww / 2, tt / 2, ln / 2, 0.36 if k < 2 else 0.6, 0.45, 32, 22)
            v = geo.transform_verts(v, geo.AXIS_Z_TO_Y)
            v = [ph_shape(*p) for p in v]
            v = [(a, b + cy, c + tt * 0.04) for a, b, c in v]
            B.add(f"{fn}_seg{k + 1}_{side}", geo.mesh_from_data("ph", v, fc), m.white, jn, mir)
            # joint barrel
            jr = tt * 0.35
            B.cyl(f"{fn}_joint{k + 1}_{side}", jn, (0, 0, -tt * 0.05), jr, ww * 1.02, axis="X", mat=m.graphite, bevel=0.0008, segs=28, mirror=mir)
            B.cyl(f"{fn}_pin{k + 1}_{side}", jn, (0, 0, -tt * 0.05), jr * 0.45, ww * 1.08, axis="X", mat=m.chrome, segs=20, mirror=mir)
            # palmar rubber grip pads on each segment
            pr = (ww * 0.40, ln * (0.30 if k < 2 else 0.34), tt * 0.20)
            pc = (0, cy + (0.0 if k < 2 else ln * 0.16), -tt * 0.36)
            B.add(f"{fn}_pad{k + 1}_{side}", geo.se_mesh("pd", pr[0], pr[2], pr[1], 0.6, 0.6, 24, 12, None, Matrix.Translation(pc) @ geo.AXIS_Z_TO_Y), m.rubber, jn, mir)

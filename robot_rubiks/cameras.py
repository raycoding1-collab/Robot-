"""Cinematography: physical cameras, focus pulls, operator handheld drift, cuts."""
from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector

from . import geo
from .choreography import Choreography, smooth, smooth_noise
from .rig_def import CAM_END_POS, FPS


def _look(loc, tgt, roll=0.0):
    d = Vector(tgt) - Vector(loc)
    q = d.to_track_quat("-Z", "Y")
    if roll:
        q = q @ Matrix.Rotation(roll, 4, "Z").to_quaternion()
    return q


def make_camera(name, coll, lens, fstop, sensor=36.0, blades=7):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = sensor
    cd.clip_start = 0.02
    cd.clip_end = 400
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = fstop
    cd.dof.aperture_blades = blades
    cd.dof.aperture_rotation = math.radians(12)
    cd.dof.aperture_ratio = 1.0
    cam = bpy.data.objects.new(name, cd)
    geo.link(cam, coll)
    cam.rotation_mode = "QUATERNION"
    focus = bpy.data.objects.new(name + "_focus", None)
    focus.empty_display_size = 0.02
    geo.link(focus, coll)
    cd.dof.focus_object = focus
    return cam, focus


def shot_list(c: Choreography):
    """The edit: one dict per shot (times in seconds)."""
    T = c.T
    solve_mid = c.solve.turns[len(c.solve.turns) // 2].t0 - 0.02
    insp1 = T["insp"][1]["rot"][0] - 0.25
    shots = [
        # name, t0, t1, lens, fstop, pos(t0), pos(t1), target(t0), target(t1), focus mode
        dict(name="CAM_01_Establish", t0=0.0, t1=T["lift1"] - 0.15, lens=32, f=2.8,
             p0=(1.02, -1.95, 1.30), p1=(0.84, -1.70, 1.25), a0=(-0.04, -0.26, 0.97), a1=(-0.02, -0.28, 1.00), focus="cube"),
        dict(name="CAM_02_InspectCU", t0=T["lift1"] - 0.15, t1=insp1, lens=85, f=2.4,
             p0=(0.47, -1.02, 1.06), p1=(0.40, -0.92, 1.09), a0=(0.0, -0.27, 1.12), a1=(0.0, -0.25, 1.17), focus="cube"),
        dict(name="CAM_03_Profile", t0=insp1, t1=T["insp_end"] + 0.05, lens=65, f=2.2,
             p0=(-0.86, -0.70, 1.30), p1=(-0.80, -0.60, 1.29), a0=(0.0, -0.20, 1.22), a1=(0.0, -0.19, 1.22), focus="mix"),
        dict(name="CAM_04_Face", t0=T["insp_end"] + 0.05, t1=T["solve0"] + 0.15, lens=75, f=2.4,
             p0=(0.14, -1.08, 1.27), p1=(0.12, -1.00, 1.28), a0=(0.0, -0.10, 1.28), a1=(0.0, -0.10, 1.27), focus="head"),
        dict(name="CAM_05_SolveA", t0=T["solve0"] + 0.15, t1=solve_mid, lens=100, f=4.0,
             p0=(0.36, -0.86, 1.03), p1=(0.33, -0.83, 1.035), a0=(0.0, -0.30, 1.00), a1=(0.0, -0.30, 1.00), focus="cube"),
        dict(name="CAM_06_SolveB", t0=solve_mid, t1=T["solved"] + 0.40, lens=58, f=3.2,
             p0=(-0.44, -0.62, 1.30), p1=(-0.42, -0.60, 1.29), a0=(0.0, -0.30, 1.00), a1=(0.0, -0.30, 1.0), focus="cube"),
        dict(name="CAM_07_Finale", t0=T["solved"] + 0.40, t1=c.duration + 0.1, lens=42, f=2.8,
             p0=tuple(CAM_END_POS + Vector((0.06, -0.10, 0.02))), p1=tuple(CAM_END_POS), a0=(-0.02, -0.25, 1.05), a1=(0.0, -0.18, 1.12), focus="end"),
    ]
    return shots


def camera_at(shots, t):
    """(position, target, lens) of the cut camera at time t (without handheld drift)."""
    for s in shots:
        if s["t0"] <= t < s["t1"] or s is shots[-1]:
            u = smooth((t - s["t0"]) / max(1e-3, s["t1"] - s["t0"]))
            return Vector(s["p0"]).lerp(Vector(s["p1"]), u), Vector(s["a0"]).lerp(Vector(s["a1"]), u), s["lens"]


def build_shots(c: Choreography, coll, head_world_fn):
    """head_world_fn(t) -> world position between the eyes (for focus pulls)."""
    T = c.T
    shots = shot_list(c)
    scene = bpy.context.scene
    cams = []
    for k, s in enumerate(shots):
        cam, focus = make_camera(s["name"], coll, s["lens"], s["f"])
        cams.append((s, cam, focus))
        # bind camera to timeline marker for the cut
        mk = scene.timeline_markers.new(s["name"], frame=int(round(s["t0"] * FPS)) + 1)
        mk.camera = cam
        f0 = int(math.floor(s["t0"] * FPS)) - 1
        f1 = int(math.ceil(s["t1"] * FPS)) + 2
        for fr in range(max(1, f0), f1 + 1):
            t = (fr - 1) / FPS
            u = smooth((t - s["t0"]) / max(1e-3, s["t1"] - s["t0"]))
            # operator: slow eased move + handheld micro-drift (breathing, weight shifts)
            pos = Vector(s["p0"]).lerp(Vector(s["p1"]), u)
            tgt = Vector(s["a0"]).lerp(Vector(s["a1"]), u)
            amp = 0.0016 if s["lens"] < 70 else 0.0009
            pos += Vector((smooth_noise(t, 100 + k), smooth_noise(t, 200 + k), smooth_noise(t, 300 + k))) * amp
            rot_n = Vector((smooth_noise(t, 400 + k, (0.4, 0.9, 1.9, 3.1)), smooth_noise(t, 500 + k, (0.35, 0.8, 1.7, 2.9)), 0)) * (0.0016 * 50 / s["lens"])
            q = _look(pos, tgt, roll=0.004 * smooth_noise(t, 600 + k, (0.2, 0.5)))
            q = q @ Matrix.Rotation(rot_n.x, 4, "X").to_quaternion() @ Matrix.Rotation(rot_n.y, 4, "Y").to_quaternion()
            cam.location = pos
            cam.rotation_quaternion = q
            cam.keyframe_insert("location", frame=fr)
            cam.keyframe_insert("rotation_quaternion", frame=fr)
            # focus
            cube_p = c.cube_world(t).translation
            head_p = head_world_fn(t)
            if s["focus"] == "cube":
                fp = cube_p
            elif s["focus"] == "head":
                fp = head_p
            elif s["focus"] == "mix":
                # rack focus: cube -> visor -> cube
                w = 0.5 - 0.5 * math.cos(2 * math.pi * min(1.0, max(0.0, (t - s["t0"]) / (s["t1"] - s["t0"]))))
                fp = cube_p.lerp(head_p, smooth(w))
            else:  # end: cube while presenting, then the face when it looks at us
                w = smooth((t - (T["look_cam"] - 0.2)) / 0.8)
                fp = cube_p.lerp(head_p, w)
            focus.location = fp
            focus.keyframe_insert("location", frame=fr)
    scene.camera = cams[0][1]
    return cams

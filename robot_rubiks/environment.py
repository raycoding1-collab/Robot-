"""Photoreal set: a modern robotics lab in late-afternoon daylight.

Lighting is physically motivated: a physical sky + sun through a glazed corner,
linear LED ceiling fixtures, and a soft bounce on the camera side (as a film
crew would use). No fake rim lights.
"""
from __future__ import annotations

import math
import random

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

from . import geo
from . import materials as M
from .rig_def import TABLE_TOP_Z

ROOM_X = (-4.2, 4.0)
ROOM_Y = (-4.6, 3.0)
ROOM_H = 3.2


# ----------------------------------------------------------------------------- materials
def _mat(name, color, rough, metal=0.0, coat=0.0, spec=0.5):
    m, nb = M.new_material(name)
    if nb is None:
        return m
    b = M.principled(nb)
    b.inputs["Base Color"].default_value = (*M.srgb_to_lin(color), 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Specular IOR Level"].default_value = spec
    return m


def linoleum():
    m, nb = M.new_material("Env_DeskLinoleum")
    if nb is None:
        return m
    b = M.principled(nb)
    tc = nb.node("ShaderNodeTexCoord")
    n1 = nb.node("ShaderNodeTexNoise", Scale=900.0, Detail=3.0, Roughness=0.5)
    n2 = nb.node("ShaderNodeTexNoise", Scale=4.0, Detail=6.0, Roughness=0.6)  # smudges
    nb.link(tc.outputs["Object"], n1.inputs["Vector"])
    nb.link(tc.outputs["Object"], n2.inputs["Vector"])
    ramp = nb.node("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*M.srgb_to_lin((0.075, 0.078, 0.082)), 1)
    ramp.color_ramp.elements[1].color = (*M.srgb_to_lin((0.105, 0.108, 0.112)), 1)
    nb.link(n1.outputs["Fac"], ramp.inputs["Fac"])
    nb.link(ramp.outputs["Color"], b.inputs["Base Color"])
    mr = nb.node("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.35
    mr.inputs["From Max"].default_value = 0.65
    mr.inputs["To Min"].default_value = 0.42
    mr.inputs["To Max"].default_value = 0.62
    nb.link(n2.outputs["Fac"], mr.inputs["Value"])
    nb.link(mr.outputs["Result"], b.inputs["Roughness"])
    bp = nb.node("ShaderNodeBump", Strength=0.04, Distance=0.0002)
    nb.link(n1.outputs["Fac"], bp.inputs["Height"])
    nb.link(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def window_glass():
    """Clear glass that lets light through for shadow/diffuse rays (no caustic blocking)."""
    m, nb = M.new_material("Env_WindowGlass")
    if nb is None:
        return m
    nt = m.node_tree
    out = nt.nodes.get("Material Output")
    b = M.principled(nb)
    b.inputs["Base Color"].default_value = (1, 1, 1, 1)
    b.inputs["Roughness"].default_value = 0.0
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["IOR"].default_value = 1.45
    lp = nb.node("ShaderNodeLightPath")
    tr = nb.node("ShaderNodeBsdfTransparent")
    mx = nb.node("ShaderNodeMixShader")
    mxf = nb.node("ShaderNodeMath", _operation="MAXIMUM")
    nb.link(lp.outputs["Is Shadow Ray"], mxf.inputs[0])
    nb.link(lp.outputs["Is Diffuse Ray"], mxf.inputs[1])
    nb.link(mxf.outputs[0], mx.inputs["Fac"])
    nb.link(b.outputs["BSDF"], mx.inputs[1])
    nb.link(tr.outputs["BSDF"], mx.inputs[2])
    nb.link(mx.outputs["Shader"], out.inputs["Surface"])
    return m


def plaster(name, color):
    m, nb = M.new_material(name)
    if nb is None:
        return m
    b = M.principled(nb)
    tc = nb.node("ShaderNodeTexCoord")
    n1 = nb.node("ShaderNodeTexNoise", Scale=3.0, Detail=8.0, Roughness=0.6)
    nb.link(tc.outputs["Object"], n1.inputs["Vector"])
    ramp = nb.node("ShaderNodeValToRGB")
    c0 = M.srgb_to_lin(tuple(x * 0.93 for x in color))
    c1 = M.srgb_to_lin(color)
    ramp.color_ramp.elements[0].color = (*c0, 1)
    ramp.color_ramp.elements[1].color = (*c1, 1)
    nb.link(n1.outputs["Fac"], ramp.inputs["Fac"])
    nb.link(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.88
    n2 = nb.node("ShaderNodeTexNoise", Scale=300.0, Detail=2.0)
    nb.link(tc.outputs["Object"], n2.inputs["Vector"])
    bp = nb.node("ShaderNodeBump", Strength=0.08, Distance=0.001)
    nb.link(n2.outputs["Fac"], bp.inputs["Height"])
    nb.link(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def screen(name, base_rgb, strength, scale=(40.0, 3.0)):
    """Emissive display with UI-like line structure (unreadable when defocused)."""
    m, nb = M.new_material(name)
    if nb is None:
        return m
    b = M.principled(nb)
    tc = nb.node("ShaderNodeTexCoord")
    mp = nb.node("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale[1], scale[0], 1)
    nb.link(tc.outputs["UV"], mp.inputs["Vector"])
    br = nb.node("ShaderNodeTexBrick")
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Mortar Size"].default_value = 0.18
    br.inputs["Color1"].default_value = (*base_rgb, 1)
    br.inputs["Color2"].default_value = tuple(c * 0.55 for c in base_rgb) + (1,)
    br.inputs["Mortar"].default_value = (0.01, 0.012, 0.016, 1)
    nb.link(mp.outputs["Vector"], br.inputs["Vector"])
    nb.link(br.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = strength
    b.inputs["Base Color"].default_value = (0.005, 0.005, 0.006, 1)
    b.inputs["Roughness"].default_value = 0.08
    b.inputs["Coat Weight"].default_value = 1.0
    return m


def box(name, size, loc, mat, coll, rot=(0, 0, 0), bevel=0.004, segs=2):
    me = geo.rounded_box_mesh(name, size[0], size[1], size[2], bevel, segs)
    o = geo.new_object(name, me, mat, collection=coll, sharp_angle=35)
    o.location = loc
    o.rotation_euler = rot
    return o


def plane(name, size, loc, rot, mat, coll):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    bmesh.ops.scale(bm, vec=(size[0], size[1], 1), verts=bm.verts)
    me = geo.mesh_from_bmesh(name, bm)
    me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            v = me.vertices[me.loops[li].vertex_index].co
            me.uv_layers[0].data[li].uv = (v.x / size[0] + 0.5, v.y / size[1] + 0.5)
    o = geo.new_object(name, me, mat, collection=coll, smooth=False)
    o.location = loc
    o.rotation_euler = rot
    return o


def cyl(name, r, h, loc, mat, coll, axis="Z", bevel=0.0, segs=48, rot=(0, 0, 0)):
    me = geo.cylinder_mesh(name, r, h, segs=segs, axis=axis, bevel=bevel)
    o = geo.new_object(name, me, mat, collection=coll, sharp_angle=35)
    o.location = loc
    o.rotation_euler = rot
    return o


# ----------------------------------------------------------------------------- build
def build_environment(coll):
    rng = random.Random(4)
    concrete = M.concrete_floor()
    wall = plaster("Env_WallPlaster", (0.62, 0.60, 0.57))
    wall_dark = plaster("Env_WallAccent", (0.20, 0.205, 0.21))
    ceiling = plaster("Env_Ceiling", (0.80, 0.80, 0.79))
    alu_black = _mat("Env_AluBlack", (0.03, 0.03, 0.032), 0.35, 1.0)
    alu = _mat("Env_Aluminium", (0.62, 0.63, 0.65), 0.28, 1.0)
    steel_pc = _mat("Env_PowderCoat", (0.06, 0.062, 0.066), 0.45, 0.0, 0.2)
    felt = _mat("Env_Felt", (0.18, 0.19, 0.20), 0.95, spec=0.2)
    glass = window_glass()
    desk = linoleum()
    mug_mat = _mat("Env_MugGlaze", (0.88, 0.87, 0.84), 0.08, 0.0, 0.6)
    paper = _mat("Env_Paper", (0.85, 0.84, 0.80), 0.8, spec=0.3)
    notebook = _mat("Env_NotebookCover", (0.05, 0.06, 0.08), 0.6, spec=0.3)
    plastic_cols = [_mat(f"Env_Bin{i}", c, 0.45) for i, c in enumerate([(0.75, 0.18, 0.12), (0.12, 0.30, 0.65), (0.85, 0.65, 0.12), (0.25, 0.27, 0.30), (0.12, 0.5, 0.35)])]
    cardboard = _mat("Env_Cardboard", (0.52, 0.40, 0.27), 0.85, spec=0.3)
    led = M.emitter("Env_LEDPanel", (1.0, 0.97, 0.92), 6.0)
    warm_bulb = M.emitter("Env_WarmBulb", (1.0, 0.62, 0.30), 30.0)
    mon = screen("Env_MonitorUI", (0.10, 0.22, 0.42), 2.2)
    lap = screen("Env_LaptopUI", (0.20, 0.30, 0.45), 1.6, (60, 4))
    rubber = M.rubber()

    # --- shell of the room
    x0, x1 = ROOM_X
    y0, y1 = ROOM_Y
    plane("Floor", (x1 - x0 + 2, y1 - y0 + 2), ((x0 + x1) / 2, (y0 + y1) / 2, 0), (0, 0, 0), concrete, coll)
    plane("Ceiling", (x1 - x0, y1 - y0), ((x0 + x1) / 2, (y0 + y1) / 2, ROOM_H), (math.pi, 0, 0), ceiling, coll)
    box("Wall_Left", (0.2, y1 - y0, ROOM_H), (x0 - 0.1, (y0 + y1) / 2, ROOM_H / 2), wall, coll, bevel=0.0)
    box("Wall_Front", (x1 - x0, 0.2, ROOM_H), ((x0 + x1) / 2, y0 - 0.1, ROOM_H / 2), wall, coll, bevel=0.0)
    # back wall: solid on the left, glazed on the right (corner window)
    gx = 0.9
    box("Wall_Back", (gx - x0, 0.2, ROOM_H), ((x0 + gx) / 2, y1 + 0.1, ROOM_H / 2), wall_dark, coll, bevel=0.0)
    # glazed corner: back (y=y1, x in [gx, x1]) and right (x=x1, y in [-2.6, y1])
    sill, head = 0.45, 2.85
    for name, a, b_, along in (("Back", gx, x1, "x"), ("Right", -2.6, y1, "y")):
        L = b_ - a
        if along == "x":
            box(f"Sill_{name}", (L, 0.25, sill), ((a + b_) / 2, y1 + 0.12, sill / 2), wall_dark, coll, bevel=0.0)
            box(f"Head_{name}", (L, 0.25, ROOM_H - head), ((a + b_) / 2, y1 + 0.12, (ROOM_H + head) / 2), wall_dark, coll, bevel=0.0)
            plane(f"Glass_{name}", (L, head - sill), ((a + b_) / 2, y1 + 0.06, (sill + head) / 2), (math.pi / 2, 0, 0), glass, coll)
        else:
            box(f"Sill_{name}", (0.25, L, sill), (x1 + 0.12, (a + b_) / 2, sill / 2), wall, coll, bevel=0.0)
            box(f"Head_{name}", (0.25, L, ROOM_H - head), (x1 + 0.12, (a + b_) / 2, (ROOM_H + head) / 2), wall, coll, bevel=0.0)
            plane(f"Glass_{name}", (L, head - sill), (x1 + 0.06, (a + b_) / 2, (sill + head) / 2), (math.pi / 2, 0, math.pi / 2), glass, coll)
        n = max(2, int(L / 1.35))
        for i in range(n + 1):
            p = a + L * i / n
            loc = (p, y1 + 0.06, (sill + head) / 2) if along == "x" else (x1 + 0.06, p, (sill + head) / 2)
            box(f"Mullion_{name}_{i}", (0.055, 0.09, head - sill) if along == "x" else (0.09, 0.055, head - sill), loc, alu_black, coll, bevel=0.003)
        for z in (sill + 0.02, head - 0.02):
            loc = ((a + b_) / 2, y1 + 0.06, z) if along == "x" else (x1 + 0.06, (a + b_) / 2, z)
            box(f"Transom_{name}_{z:.2f}", (L, 0.09, 0.05) if along == "x" else (0.09, L, 0.05), loc, alu_black, coll, bevel=0.003)
    box("Wall_Right_Solid", (0.2, -2.6 - y0, ROOM_H), (x1 + 0.1, (y0 - 2.6) / 2, ROOM_H / 2), wall, coll, bevel=0.0)
    # skirting
    box("Skirting_L", (0.02, y1 - y0, 0.08), (x0 + 0.01, (y0 + y1) / 2, 0.04), steel_pc, coll, bevel=0.001)

    # --- outside: distant buildings & trees for parallax through the glass
    city = _mat("Env_City", (0.70, 0.71, 0.72), 0.7)
    for i in range(18):
        w = rng.uniform(8, 22)
        h = rng.uniform(12, 55)
        ang = math.radians(rng.uniform(-60, 75))
        d = rng.uniform(60, 140)
        bo = box(f"Bldg_{i}", (w, w * rng.uniform(0.6, 1.2), h), (math.cos(ang) * d + 20, math.sin(ang) * d + 30, h / 2 - 4), city, coll, rot=(0, 0, rng.uniform(0, 3)), bevel=0.0)
        bo.visible_shadow = False
    ground = _mat("Env_Ground", (0.30, 0.31, 0.29), 0.9)
    plane("OutsideGround", (600, 600), (0, 0, -3.5), (0, 0, 0), ground, coll)

    # --- ceiling LED fixtures (visible geometry + area lights)
    for i, y in enumerate((-2.4, -0.6, 1.2)):
        for j, x in enumerate((-1.4, 1.4)):
            box(f"LEDHousing_{i}{j}", (1.45, 0.11, 0.05), (x, y, ROOM_H - 0.40), alu, coll, bevel=0.004)
            box(f"LEDDiffuser_{i}{j}", (1.40, 0.085, 0.004), (x, y, ROOM_H - 0.428), led, coll, bevel=0.001)
            cyl(f"LEDWire_{i}{j}a", 0.0012, 0.38, (x - 0.6, y, ROOM_H - 0.19), steel_pc, coll, segs=8)
            cyl(f"LEDWire_{i}{j}b", 0.0012, 0.38, (x + 0.6, y, ROOM_H - 0.19), steel_pc, coll, segs=8)
            ld = bpy.data.lights.new(f"LED_{i}{j}", "AREA")
            ld.shape = "RECTANGLE"
            ld.size = 1.4
            ld.size_y = 0.085
            ld.energy = 32.0
            ld.color = (1.0, 0.95, 0.88)
            ld.spread = math.radians(120)
            lo = bpy.data.objects.new(f"LED_{i}{j}", ld)
            geo.link(lo, coll)
            lo.location = (x, y, ROOM_H - 0.432)

    # --- desk
    top_t = 0.028
    box("Desk_Top", (1.60, 0.80, top_t), (0.0, -0.585, TABLE_TOP_Z - top_t / 2), desk, coll, bevel=0.004, segs=3)
    for sx in (-1, 1):
        box(f"Desk_Leg_{sx}", (0.05, 0.70, 0.04), (sx * 0.72, -0.585, 0.02), steel_pc, coll, bevel=0.004)
        box(f"Desk_Post_{sx}", (0.05, 0.05, TABLE_TOP_Z - top_t), (sx * 0.72, -0.585, (TABLE_TOP_Z - top_t) / 2), steel_pc, coll, bevel=0.004)
        box(f"Desk_Rail_{sx}", (0.04, 0.70, 0.05), (sx * 0.72, -0.585, TABLE_TOP_Z - top_t - 0.025), steel_pc, coll, bevel=0.004)
    box("Desk_Modesty", (1.40, 0.012, 0.30), (0.0, -0.95, TABLE_TOP_Z - 0.20), steel_pc, coll, bevel=0.002)

    # --- props on the desk
    # laptop (open)
    lp = bpy.data.objects.new("Laptop", None)
    geo.link(lp, coll)
    lp.location = (0.56, -0.66, TABLE_TOP_Z)
    lp.rotation_euler = (0, 0, math.radians(-28))
    b0 = box("Laptop_Base", (0.312, 0.218, 0.0145), (0, 0, 0.0073), alu, coll, bevel=0.004, segs=3)
    b0.parent = lp
    kb = box("Laptop_Keys", (0.27, 0.10, 0.001), (0, 0.035, 0.0146), alu_black, coll, bevel=0.0005)
    kb.parent = lp
    tp = box("Laptop_Pad", (0.11, 0.07, 0.0006), (0, -0.065, 0.0146), alu_black, coll, bevel=0.0003)
    tp.parent = lp
    hinge = bpy.data.objects.new("Laptop_Hinge", None)
    geo.link(hinge, coll)
    hinge.parent = lp
    hinge.location = (0, 0.105, 0.013)
    hinge.rotation_euler = (math.radians(-108), 0, 0)
    lid = box("Laptop_Lid", (0.312, 0.215, 0.006), (0, -0.1075, 0.003), alu, coll, bevel=0.003, segs=3)
    lid.parent = hinge
    scr = plane("Laptop_Screen", (0.29, 0.185), (0, -0.108, -0.0002), (math.pi, 0, 0), lap, coll)
    scr.parent = hinge
    # mug
    mg = cyl("Mug", 0.041, 0.095, (-0.56, -0.44, TABLE_TOP_Z + 0.0475), mug_mat, coll, bevel=0.003)
    inner = cyl("Mug_Coffee", 0.037, 0.002, (-0.56, -0.44, TABLE_TOP_Z + 0.082), _mat("Env_Coffee", (0.09, 0.05, 0.03), 0.05, coat=1.0), coll)
    hd = geo.torus_mesh("MugHandle", 0.026, 0.0065, 40, 10, "Y")
    ho = geo.new_object("Mug_Handle", hd, mug_mat, collection=coll)
    ho.location = (-0.56 - 0.045, -0.44, TABLE_TOP_Z + 0.05)
    ho.scale = (0.75, 1, 1)
    # notebook + pen
    nbk = box("Notebook", (0.148, 0.21, 0.012), (-0.40, -0.80, TABLE_TOP_Z + 0.006), notebook, coll, rot=(0, 0, math.radians(12)), bevel=0.002)
    pg = box("Notebook_Pages", (0.144, 0.205, 0.0105), (-0.40, -0.80, TABLE_TOP_Z + 0.006), paper, coll, rot=(0, 0, math.radians(12)), bevel=0.001)
    pg.location.x += 0.0025
    pen = cyl("Pen", 0.0045, 0.14, (-0.36, -0.79, TABLE_TOP_Z + 0.0165), alu_black, coll, axis="Y", bevel=0.001, rot=(0, 0, math.radians(20)))
    # tiny screws + hex key near the robot's left hand side
    screw = _mat("Env_Screw", (0.75, 0.75, 0.76), 0.25, 1.0)
    for i in range(5):
        x, y = 0.30 + rng.uniform(-0.04, 0.04), -0.33 + rng.uniform(-0.03, 0.03)
        s = cyl(f"Screw_{i}", 0.0017, 0.010, (x, y, TABLE_TOP_Z + 0.0017), screw, coll, axis="X", segs=12, rot=(0, 0, rng.uniform(0, 3.1)))
        cyl(f"ScrewHead_{i}", 0.0032, 0.0018, (0, 0, 0), screw, coll, axis="X", segs=16).parent = s
    hexk = geo.curve_tube("HexKey", [(0.36, -0.42, TABLE_TOP_Z + 0.0022), (0.42, -0.40, TABLE_TOP_Z + 0.0022), (0.425, -0.398, TABLE_TOP_Z + 0.006), (0.425, -0.398, TABLE_TOP_Z + 0.022)], 0.0022, alu_black, None, coll)
    # cable coil
    pts = []
    for k in range(60):
        a = k * 0.42
        r = 0.045 + 0.004 * math.sin(k * 0.7)
        pts.append((0.68 + r * math.cos(a) + k * 0.0006, -0.34 + r * math.sin(a), TABLE_TOP_Z + 0.0025 + 0.0045 * (k % 9) / 9))
    geo.curve_tube("Cable", pts, 0.0024, rubber, None, coll, resolution=4)

    # --- background: workbench, robot arm, monitor, shelving, pendant bulbs
    box("Bench_Top", (2.6, 0.75, 0.04), (-1.6, 2.45, 0.90), _mat("Env_Butcher", (0.55, 0.40, 0.26), 0.55), coll, bevel=0.004)
    for x in (-2.8, -0.4):
        box(f"Bench_Leg_{x}", (0.05, 0.65, 0.88), (x, 2.45, 0.44), steel_pc, coll, bevel=0.004)
    box("Pegboard", (2.4, 0.02, 1.0), (-1.6, 2.88, 1.62), felt, coll, bevel=0.002)
    for i in range(14):
        box(f"Tool_{i}", (rng.uniform(0.02, 0.05), 0.02, rng.uniform(0.12, 0.28)), (-2.7 + i * 0.16, 2.85, 1.62 + rng.uniform(-0.25, 0.25)), rng.choice([alu_black, alu] + plastic_cols[:2]), coll, bevel=0.003)
    box("Monitor", (0.62, 0.03, 0.37), (-1.0, 2.70, 1.25), alu_black, coll, bevel=0.006)
    plane("Monitor_Screen", (0.60, 0.35), (-1.0, 2.684, 1.25), (math.pi / 2, 0, 0), mon, coll)
    cyl("Monitor_Stand", 0.015, 0.30, (-1.0, 2.72, 1.07), alu, coll)
    # industrial arm on the bench
    arm_white = _mat("Env_ArmWhite", (0.78, 0.78, 0.76), 0.35, coat=0.3)
    cyl("Arm_Base", 0.09, 0.08, (-2.2, 2.40, 0.96), steel_pc, coll, bevel=0.01)
    cyl("Arm_J1", 0.07, 0.16, (-2.2, 2.40, 1.08), arm_white, coll, bevel=0.01)
    cyl("Arm_Link1", 0.05, 0.42, (-2.12, 2.40, 1.36), arm_white, coll, bevel=0.01, rot=(0, math.radians(22), 0))
    cyl("Arm_J2", 0.06, 0.13, (-2.04, 2.40, 1.56), steel_pc, coll, axis="Y", bevel=0.01)
    cyl("Arm_Link2", 0.042, 0.36, (-1.88, 2.40, 1.48), arm_white, coll, bevel=0.01, rot=(0, math.radians(115), 0))
    # boxes and bins on the bench
    for i in range(6):
        box(f"BenchBox_{i}", (rng.uniform(0.15, 0.32), rng.uniform(0.15, 0.3), rng.uniform(0.08, 0.22)), (-2.85 + i * 0.42, 2.30 + rng.uniform(-0.05, 0.1), 0.92 + 0.05), rng.choice(plastic_cols + [cardboard]), coll, bevel=0.006)
    # shelving at the back-left
    for z in (0.3, 0.75, 1.2, 1.65, 2.1):
        box(f"Shelf_{z}", (1.2, 0.45, 0.025), (-3.55, 1.1, z), steel_pc, coll, bevel=0.002)
        for i in range(3):
            if rng.random() < 0.8:
                box(f"Bin_{z}_{i}", (0.34, 0.40, rng.uniform(0.16, 0.3)), (-3.95 + i * 0.4, 1.1, z + 0.1), rng.choice(plastic_cols + [cardboard]), coll, bevel=0.01)
    for x in (-4.12, -2.98):
        for y in (0.9, 1.3):
            box(f"ShelfPost_{x}_{y}", (0.03, 0.03, 2.3), (x, y, 1.15), steel_pc, coll, bevel=0.002)
    # warm pendant bulbs (bokeh)
    for i, (x, y, z) in enumerate([(-0.9, 1.9, 2.25), (-2.1, 1.85, 2.35), (0.4, 2.2, 2.30)]):
        cyl(f"Pendant_Cord_{i}", 0.002, ROOM_H - z, (x, y, (ROOM_H + z) / 2), steel_pc, coll, segs=8)
        b = geo.se_mesh(f"Bulb_{i}", 0.032, 0.032, 0.045, 1, 1, 24, 12)
        bo = geo.new_object(f"Pendant_Bulb_{i}", b, warm_bulb, collection=coll)
        bo.location = (x, y, z - 0.05)
    # robot stool
    cyl("Stool_Seat", 0.21, 0.06, (0, 0.03, 0.47), steel_pc, coll, bevel=0.015)
    cyl("Stool_Cushion", 0.20, 0.02, (0, 0.03, 0.505), rubber, coll, bevel=0.008)
    cyl("Stool_Post", 0.03, 0.44, (0, 0.03, 0.22), alu, coll)
    cyl("Stool_Base", 0.26, 0.018, (0, 0.03, 0.01), steel_pc, coll, bevel=0.006)


def setup_world(sun_elevation=17.0, sun_azimuth=52.0):
    """Physical sky + matching sun lamp.

    sun_azimuth: degrees from +Y toward +X, i.e. the sun stands outside the glazed
    back/right corner and rakes in past the robot's left shoulder.
    """
    w = bpy.data.worlds.new("LabSky")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes.get("Background")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False
    sky.sun_elevation = math.radians(sun_elevation)
    # Blender's sky: rotation 0 -> sun towards -Y; increases counter-clockwise
    sky.sun_rotation = math.radians(180.0 - sun_azimuth)
    sky.altitude = 120.0
    sky.air_density = 1.0
    sky.aerosol_density = 1.6
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.075
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 5.5
    sun.angle = math.radians(0.6)
    sun.color = (1.0, 0.86, 0.70)
    so = bpy.data.objects.new("Sun", sun)
    geo.link(so)
    az = math.radians(sun_azimuth)
    el = math.radians(sun_elevation)
    d = Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))  # towards the sun
    so.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    return w, so


def add_bounce(coll):
    """Large soft bounce on the camera side, as used on set (motivated by the windows)."""
    l = bpy.data.lights.new("Bounce", "AREA")
    l.shape = "RECTANGLE"
    l.size = 2.6
    l.size_y = 2.0
    l.energy = 110.0
    l.color = (1.0, 0.96, 0.92)
    o = bpy.data.objects.new("Bounce", l)
    geo.link(o, coll)
    o.location = (1.9, -2.3, 1.75)
    o.visible_glossy = False  # light the subject, but no tell-tale rectangle in reflections
    o.rotation_euler = (Vector((0.0, -0.25, 1.05)) - o.location).to_track_quat("-Z", "Y").to_euler()
    l2 = bpy.data.lights.new("Fill_Neg", "AREA")
    l2.shape = "DISK"
    l2.size = 1.2
    l2.energy = 18.0
    l2.color = (0.92, 0.95, 1.0)
    o2 = bpy.data.objects.new("Fill", l2)
    geo.link(o2, coll)
    o2.location = (-1.7, -1.4, 1.2)
    o2.visible_glossy = False
    o2.rotation_euler = (Vector((0.0, -0.2, 1.0)) - o2.location).to_track_quat("-Z", "Y").to_euler()
    return o, o2

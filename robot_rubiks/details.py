"""Manufacturing details: fasteners and printed markings, ray-cast onto the shells."""
from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import geo
from . import materials as M


def _bvh(obj):
    me = obj.data
    return BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])


def _surface(obj, origin, direction):
    hit = _bvh(obj).ray_cast(Vector(origin), Vector(direction).normalized())
    if hit[0] is None:
        return None, None
    loc, nrm = hit[0], hit[1]
    if nrm.dot(Vector(direction)) > 0:
        nrm = -nrm
    return loc, nrm


def _orient(normal, up_hint=Vector((0, 0, 1))):
    z = normal.normalized()
    up = up_hint - z * up_hint.dot(z)
    if up.length < 1e-4:
        up = Vector((0, 1, 0)) - z * z.y
    y = up.normalized()
    x = y.cross(z)
    return Matrix((x, y, z)).transposed().to_4x4()


def _screw_mesh(r=0.0021):
    head = geo.cylinder_mesh("screw", r, 0.0007, segs=24, axis="Z", bevel=0.00025, bevel_segs=2, center=(0, 0, 0.0001))
    return head


def add_screw(part, origin, direction, mats, name, r=0.0021, coll=None):
    loc, nrm = _surface(part, origin, direction)
    if loc is None:
        return None
    me = _screw_mesh(r)
    o = geo.new_object(name, me, mats.graphite, parent=part.parent, collection=coll, sharp_angle=35)
    o.matrix_parent_inverse = Matrix.Identity(4)
    o.matrix_basis = Matrix.Translation(loc) @ _orient(nrm)
    # hex socket (dark recess)
    hx = geo.cylinder_mesh("hex", r * 0.42, 0.0002, segs=6, axis="Z", center=(0, 0, 0.00046))
    h = geo.new_object(name + "_hex", hx, mats.face, parent=o, collection=coll, smooth=False)
    return o


def add_label(part, origin, direction, text, size, mat, name, coll=None, up=Vector((0, 0, 1)), extrude=0.00003):
    loc, nrm = _surface(part, origin, direction)
    if loc is None:
        return None
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = text
    cu.size = size
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.extrude = extrude
    cu.space_character = 1.08
    o = bpy.data.objects.new(name, cu)
    geo.link(o, coll)
    cu.materials.append(mat)
    o.parent = part.parent
    o.matrix_parent_inverse = Matrix.Identity(4)
    o.matrix_basis = Matrix.Translation(loc + nrm * 0.00012) @ _orient(nrm, up)
    return o


def add_details(parts, mats, coll):
    P = {p.name: p for p in parts}
    ink = M.new_material("Robot_PrintInk")[0]
    if ink.node_tree and ink.node_tree.nodes.get("Principled BSDF"):
        b = ink.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.035, 0.037, 0.04, 1)
        b.inputs["Roughness"].default_value = 0.55
    grey_ink = M.new_material("Robot_PrintGrey")[0]
    b = grey_ink.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.25, 0.26, 0.27, 1)
    b.inputs["Roughness"].default_value = 0.5
    n = 0
    # chest plates: four corner screws each
    for side, sx in (("R", -1), ("L", 1)):
        part = P.get(f"chest_plate_{side}")
        if part:
            for (x, z) in ((0.035, 0.300), (0.130, 0.270), (0.040, 0.140), (0.128, 0.150)):
                if add_screw(part, (sx * x, -0.4, z), (0, 1, 0), mats, f"screw_chest_{side}_{n}", coll=coll):
                    n += 1
        rib = P.get(f"chest_rib_{side}")
        if rib:
            for (x, z) in ((0.05, 0.085), (0.11, 0.085)):
                add_screw(rib, (sx * x, -0.4, z), (0, 1, 0), mats, f"screw_rib_{side}_{x}", r=0.0018, coll=coll)
    # model designation + serial on the chest
    if "chest_plate_L" in P:
        add_label(P["chest_plate_L"], (0.085, -0.4, 0.232), (0, 1, 0), "AX-07", 0.0125, ink, "label_model", coll)
        add_label(P["chest_plate_L"], (0.085, -0.4, 0.218), (0, 1, 0), "HUMANOID  RESEARCH  UNIT", 0.0034, grey_ink, "label_sub", coll)
    if "chest_plate_R" in P:
        add_label(P["chest_plate_R"], (-0.10, -0.4, 0.165), (0, 1, 0), "⚠ 48V", 0.0055, grey_ink, "label_volt", coll)
    # arms: shell screws + serials on the lateral side
    for side in ("R", "L"):
        lat = -1 if side == "R" else 1  # mirrored geometry: lateral is local -X for R (hinge axis), handle both
        for name, ys in ((f"upperarm_shell_{side}", (0.09, 0.19)), (f"forearm_shell_{side}", (0.05, 0.13))):
            part = P.get(name)
            if not part:
                continue
            for y in ys:
                for d in ((1, 0, 0), (-1, 0, 0)):
                    add_screw(part, (-d[0] * 0.2, y, 0.0), d, mats, f"screw_{name}_{y}_{d[0]}", r=0.0019, coll=coll)
            add_label(part, (0.0, (ys[0] + ys[1]) / 2, 0.2), (0, 0, -1), f"SN 0471-{side}", 0.0036, grey_ink, f"label_{name}", coll, up=Vector((0, 1, 0)))
        cuff = P.get(f"wrist_cuff_{side}")
        if cuff:
            add_label(cuff, (0.0, 0.020, -0.2), (0, 0, 1), "07", 0.0055, grey_ink, f"label_cuff_{side}", coll, up=Vector((0, 1, 0)))
        palm = P.get(f"palm_dorsal_{side}")
        if palm:
            for x in (-0.022, 0.022):
                add_screw(palm, (x, 0.025, 0.2), (0, 0, -1), mats, f"screw_palm_{side}_{x}", r=0.0015, coll=coll)
    # helmet: two screws below each ear pod, small marking on the brow
    for part_name in ("helmet_back",):
        part = P.get(part_name)
        if part:
            for sx in (-1, 1):
                add_screw(part, (sx * 0.3, 0.035, 0.055), (-sx, 0, 0), mats, f"screw_helmet_{sx}", r=0.0018, coll=coll)
    if "helmet_brow" in P:
        add_label(P["helmet_brow"], (0.0, -0.4, 0.178), (0, 1, 0.15), "AX", 0.0062, grey_ink, "label_brow", coll)
    # back module
    if "back_module" in P:
        for (x, z) in ((-0.07, 0.10), (0.07, 0.10), (-0.07, 0.28), (0.07, 0.28)):
            add_screw(P["back_module"], (x, 0.4, z), (0, -1, 0), mats, f"screw_back_{x}_{z}", r=0.0024, coll=coll)

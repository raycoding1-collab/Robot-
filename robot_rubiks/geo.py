"""Procedural mesh helpers (pure bpy/bmesh, no external assets)."""
from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Matrix, Vector


def _sgnpow(w: float, e: float) -> float:
    return math.copysign(abs(w) ** e, w)


def link(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def new_object(name, mesh, mat=None, parent=None, collection=None, smooth=True, sharp_angle=None):
    obj = bpy.data.objects.new(name, mesh)
    link(obj, collection)
    if mat is not None:
        if isinstance(mat, (list, tuple)):
            for m in mat:
                mesh.materials.append(m)
        else:
            mesh.materials.append(mat)
    if smooth:
        mesh.shade_smooth()
        if sharp_angle is not None:
            mesh.set_sharp_from_angle(angle=math.radians(sharp_angle))
    if parent is not None:
        obj.parent = parent
    return obj


def mesh_from_data(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate(clean_customdata=False)
    me.update()
    return me


def mesh_from_bmesh(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return me


# ----------------------------------------------------------------------------
# Superellipsoid family (rounded boxes, capsules, pods ...)
# ----------------------------------------------------------------------------
def superellipsoid(rx, ry, rz, e1=1.0, e2=1.0, nu=48, nv=24, shape=None):
    """Return (verts, faces) of a closed superellipsoid. Long axis = Z.

    e1: vertical squareness, e2: horizontal squareness (1 = round, ->0 = boxy)
    shape(x, y, z, t) -> (x, y, z) optional deformation; t = z/rz in [-1, 1].
    """
    verts = []
    faces = []
    # latitude rings excluding poles
    for j in range(1, nv):
        v = -math.pi / 2 + math.pi * j / nv
        cv, sv = math.cos(v), math.sin(v)
        for i in range(nu):
            u = -math.pi + 2 * math.pi * i / nu
            x = rx * _sgnpow(cv, e1) * _sgnpow(math.cos(u), e2)
            y = ry * _sgnpow(cv, e1) * _sgnpow(math.sin(u), e2)
            z = rz * _sgnpow(sv, e1)
            if shape:
                x, y, z = shape(x, y, z, z / rz)
            verts.append((x, y, z))
    bottom = len(verts)
    b = (0.0, 0.0, -rz)
    t = (0.0, 0.0, rz)
    if shape:
        b = shape(b[0], b[1], b[2], -1.0)
        t = shape(t[0], t[1], t[2], 1.0)
    verts.append(b)
    top = len(verts)
    verts.append(t)
    rings = nv - 1
    for j in range(rings - 1):
        for i in range(nu):
            a = j * nu + i
            bq = j * nu + (i + 1) % nu
            c = (j + 1) * nu + (i + 1) % nu
            d = (j + 1) * nu + i
            faces.append((a, bq, c, d))
    for i in range(nu):
        faces.append((bottom, (i + 1) % nu, i))
        a = (rings - 1) * nu
        faces.append((top, a + i, a + (i + 1) % nu))
    return verts, faces


def transform_verts(verts, mat: Matrix):
    return [tuple(mat @ Vector(v)) for v in verts]


AXIS_Z_TO_Y = Matrix.Rotation(-math.pi / 2, 4, "X")  # maps +Z -> +Y
AXIS_Z_TO_X = Matrix.Rotation(math.pi / 2, 4, "Y")  # maps +Z -> +X


def se_mesh(name, rx, ry, rz, e1=1.0, e2=1.0, nu=48, nv=24, shape=None, mat4=None):
    v, f = superellipsoid(rx, ry, rz, e1, e2, nu, nv, shape)
    if mat4 is not None:
        v = transform_verts(v, mat4)
    return mesh_from_data(name, v, f)


def limb_shell(name, length, r0, r1, flat=1.0, e1=0.50, e2=0.72, nu=40, nv=28, start=0.0, bulge=0.0):
    """Tapered rounded shell along +Y from `start` to `start+length`.

    r0 radius at start, r1 at end, flat = z/x radius ratio, bulge adds mid fullness.
    """

    def shape(x, y, z, t):
        s = (t + 1) / 2  # 0 at start, 1 at end
        r = r0 + (r1 - r0) * s
        b = 1.0 + bulge * math.sin(math.pi * s)
        return x * r * b, y * r * flat * b, z

    v, f = superellipsoid(1.0, 1.0, length / 2, e1, e2, nu, nv, shape)
    m = Matrix.Translation((0, start + length / 2, 0)) @ AXIS_Z_TO_Y
    v = transform_verts(v, m)
    return mesh_from_data(name, v, f)


def cylinder_mesh(name, r, depth, segs=48, axis="Z", bevel=0.0, bevel_segs=3, center=(0, 0, 0)):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r, radius2=r, depth=depth)
    if bevel > 0:
        edges = [e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) < 1e-9 and abs(abs(e.verts[0].co.z) - depth / 2) < 1e-9]
        bmesh.ops.bevel(bm, geom=edges, offset=bevel, segments=bevel_segs, profile=0.5, affect="EDGES", clamp_overlap=True)
    m = Matrix.Identity(4)
    if axis == "X":
        m = AXIS_Z_TO_X
    elif axis == "Y":
        m = AXIS_Z_TO_Y
    bmesh.ops.transform(bm, matrix=Matrix.Translation(center) @ m, verts=bm.verts)
    return mesh_from_bmesh(name, bm)


def torus_mesh(name, R, r, segs=64, rsegs=12, axis="Z", center=(0, 0, 0)):
    verts, faces = [], []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        for j in range(rsegs):
            b = 2 * math.pi * j / rsegs
            x = (R + r * math.cos(b)) * math.cos(a)
            y = (R + r * math.cos(b)) * math.sin(a)
            z = r * math.sin(b)
            verts.append((x, y, z))
    for i in range(segs):
        for j in range(rsegs):
            a = i * rsegs + j
            b = ((i + 1) % segs) * rsegs + j
            c = ((i + 1) % segs) * rsegs + (j + 1) % rsegs
            d = i * rsegs + (j + 1) % rsegs
            faces.append((a, b, c, d))
    m = Matrix.Identity(4)
    if axis == "X":
        m = AXIS_Z_TO_X
    elif axis == "Y":
        m = AXIS_Z_TO_Y
    verts = transform_verts(verts, Matrix.Translation(center) @ m)
    return mesh_from_data(name, verts, faces)


def rounded_box_bm(sx, sy, sz, bevel, segs=4):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    if bevel > 0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=segs, profile=0.5, affect="EDGES", clamp_overlap=True)
    return bm


def rounded_box_mesh(name, sx, sy, sz, bevel, segs=4, center=(0, 0, 0)):
    bm = rounded_box_bm(sx, sy, sz, bevel, segs)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    return mesh_from_bmesh(name, bm)


def rounded_rect_tile_bm(size, radius, thickness, top_bevel, corner_segs=6):
    """Flat rounded-square tile lying on z=0 (bottom) extruded to +thickness."""
    bm = bmesh.new()
    h = size / 2
    pts = []
    for cx, cy, a0 in ((h - radius, h - radius, 0), (-h + radius, h - radius, 90), (-h + radius, -h + radius, 180), (h - radius, -h + radius, 270)):
        for k in range(corner_segs + 1):
            a = math.radians(a0 + 90 * k / corner_segs)
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    bottom = [bm.verts.new((x, y, 0.0)) for x, y in pts]
    top = [bm.verts.new((x, y, thickness)) for x, y in pts]
    n = len(pts)
    bm.faces.new(list(reversed(bottom)))
    ftop = bm.faces.new(top)
    for i in range(n):
        bm.faces.new((bottom[i], bottom[(i + 1) % n], top[(i + 1) % n], top[i]))
    if top_bevel > 0:
        edges = list(ftop.edges)
        bmesh.ops.bevel(bm, geom=edges, offset=top_bevel, segments=3, profile=0.5, affect="EDGES", clamp_overlap=True)
    return bm


def bisect_keep(bm, co, no, keep_positive=True):
    """Cut the bmesh with a plane, keep the side where (p-co).no > 0."""
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=co, plane_no=no, clear_inner=keep_positive, clear_outer=not keep_positive)
    return bm


def bm_from_verts_faces(verts, faces):
    bm = bmesh.new()
    vs = [bm.verts.new(v) for v in verts]
    for f in faces:
        try:
            bm.faces.new([vs[i] for i in f])
        except ValueError:
            pass
    bm.normal_update()
    return bm


def panel_mesh(name, verts, faces, cuts):
    """Superellipsoid -> keep region inside all planes (list of (co, no) keep-positive)."""
    bm = bm_from_verts_faces(verts, faces)
    for co, no in cuts:
        bisect_keep(bm, co, no, True)
    return mesh_from_bmesh(name, bm)


def add_solidify(obj, thickness, offset=-1.0, rim=True):
    m = obj.modifiers.new("Solidify", "SOLIDIFY")
    m.thickness = thickness
    m.offset = offset
    m.use_rim = rim
    m.use_even_offset = True
    return m


def add_bevel(obj, width, segments=2, angle=40):
    m = obj.modifiers.new("Bevel", "BEVEL")
    m.width = width
    m.segments = segments
    m.limit_method = "ANGLE"
    m.angle_limit = math.radians(angle)
    m.harden_normals = False
    return m


def add_subsurf(obj, levels=1, render_levels=2):
    m = obj.modifiers.new("Subsurf", "SUBSURF")
    m.levels = levels
    m.render_levels = render_levels
    return m


def curve_tube(name, points, radius, mat=None, parent=None, collection=None, resolution=12):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 4
    cu.resolution_u = resolution
    cu.use_fill_caps = True
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(points) - 1)
    for bp, p in zip(sp.bezier_points, points):
        bp.co = p
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, cu)
    link(obj, collection)
    if mat:
        cu.materials.append(mat)
    if parent:
        obj.parent = parent
    return obj


def join_meshes(name, objs):
    """Join mesh objects into one (keeps materials). Returns new object."""
    import bpy as _b

    if len(objs) == 1:
        objs[0].name = name
        return objs[0]
    ctx = {"active_object": objs[0], "selected_editable_objects": objs, "object": objs[0]}
    with _b.context.temp_override(**ctx):
        _b.ops.object.join()
    objs[0].name = name
    return objs[0]

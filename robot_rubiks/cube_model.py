"""Geometry for a modern stickerless-style 3x3 speed cube.

Each cubie object has its origin at the cube centre, so every layer turn is a
pure rotation of the object (exact arcs, exact motion blur).
"""
from __future__ import annotations

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import geo
from . import materials as M
from .cube_logic import FACE_COLOR, FACE_NORMAL, CubeState
from .rig_def import CUBIE_BEVEL, CUBIE_PITCH, CUBIE_SIZE, TILE_RADIUS, TILE_SIZE, TILE_THICK


def _tile_matrix(normal: Vector, center: Vector):
    # tile is modelled on z=0 facing +Z; rotate +Z onto the face normal
    q = Vector((0, 0, 1)).rotation_difference(normal)
    return Matrix.Translation(center) @ q.to_matrix().to_4x4()


def build_cube(coll, name="PuzzleCube"):
    body = M.cube_body()
    tiles = {c: M.cube_tile(c) for c in set(FACE_COLOR.values())}
    root = bpy.data.objects.new(name, None)
    root.empty_display_size = 0.04
    root.empty_display_type = "CUBE"
    root.rotation_mode = "QUATERNION"
    geo.link(root, coll)

    state = CubeState()
    cubies = []
    for i, home in enumerate(state.homes):
        hv = Vector(home) * CUBIE_PITCH
        bm = geo.rounded_box_bm(CUBIE_SIZE, CUBIE_SIZE, CUBIE_SIZE, CUBIE_BEVEL, 4)
        bmesh.ops.translate(bm, vec=hv, verts=bm.verts)
        for f in bm.faces:
            f.material_index = 0
        mats = [body]
        for face, n in FACE_NORMAL.items():
            nv = Vector(n.tolist())
            if Vector(home).dot(nv) > 0.5:
                mats.append(tiles[FACE_COLOR[face]])
                idx = len(mats) - 1
                tb = geo.rounded_rect_tile_bm(TILE_SIZE, TILE_RADIUS, TILE_THICK, 0.00022, 6)
                center = hv + nv * (CUBIE_SIZE / 2 - 0.00005)
                bmesh.ops.transform(tb, matrix=_tile_matrix(nv, center), verts=tb.verts)
                for f in tb.faces:
                    f.material_index = idx
                tmp = bpy.data.meshes.new("tmp")
                tb.to_mesh(tmp)
                tb.free()
                bm.from_mesh(tmp)
                bpy.data.meshes.remove(tmp)
        me = geo.mesh_from_bmesh(f"cubie_{i:02d}", bm)
        obj = geo.new_object(f"Cubie_{home[0]:+d}{home[1]:+d}{home[2]:+d}", me, mats, parent=root, collection=coll, sharp_angle=30)
        obj.rotation_mode = "QUATERNION"
        cubies.append(obj)
    return root, cubies

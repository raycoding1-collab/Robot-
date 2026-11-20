"""Shared dimensions / skeleton definition (metres, seconds).

World: Z up, robot sits at the origin facing -Y. Robot's right side = -X.
Cube-local frame: +X=R, -Y=F (faces the solver), +Z=U. Upright cube in front of
the robot therefore has world rotation Rz(180deg).
"""
import math

from mathutils import Matrix, Vector

FPS = 24

# ---------------------------------------------------------------- puzzle cube
CUBIE_PITCH = 0.019
CUBE_HALF = 1.5 * CUBIE_PITCH  # 28.5 mm
CUBIE_SIZE = 0.0187
CUBIE_BEVEL = 0.0017
TILE_SIZE = 0.0158
TILE_RADIUS = 0.0024
TILE_THICK = 0.0004

# ---------------------------------------------------------------- environment
TABLE_TOP_Z = 0.74
CUBE_REST_POS = Vector((0.0, -0.385, TABLE_TOP_Z + CUBE_HALF))
CUBE_UPRIGHT = Matrix.Rotation(math.pi, 4, "Z")

# ---------------------------------------------------------------- skeleton
PELVIS_POS = Vector((0.0, 0.03, 0.60))
SPINE_OFF = Vector((0.0, 0.0, 0.10))
CHEST_OFF = Vector((0.0, 0.0, 0.16))
NECK_OFF = Vector((0.0, 0.006, 0.335))
HEAD_OFF = Vector((0.0, -0.004, 0.085))
SHOULDER_OFF_R = Vector((-0.205, 0.012, 0.282))  # relative to chest; left = mirrored X

UPPER_LEN = 0.285
FORE_LEN = 0.262
WRIST_ROLL_AT = 0.160

EYE_OFFSET = Vector((0.0, -0.10, 0.104))  # head-local, between the eyes

# Hand frame (canonical RIGHT hand): Y = towards fingers, Z = dorsal (back of hand),
# X = Y x Z = pinky side. Thumb is on -X.
FINGERS = {
    "index": dict(base=(-0.0265, 0.087, 0.001), splay=math.radians(-5), lens=(0.044, 0.026, 0.023), width=0.0172, thick=0.0152, couple=0.72),
    "middle": dict(base=(-0.0075, 0.092, 0.002), splay=math.radians(0), lens=(0.048, 0.029, 0.024), width=0.0176, thick=0.0156, couple=0.72),
    "ring": dict(base=(0.0110, 0.089, 0.001), splay=math.radians(5), lens=(0.045, 0.027, 0.023), width=0.0168, thick=0.0150, couple=0.72),
    "pinky": dict(base=(0.0285, 0.081, -0.001), splay=math.radians(11), lens=(0.035, 0.021, 0.020), width=0.0148, thick=0.0138, couple=0.72),
    "thumb": dict(base=(-0.0245, 0.020, -0.011), dir=(-0.60, 0.72, -0.35), nail=(-0.55, -0.05, 0.85), lens=(0.042, 0.031, 0.027), width=0.0195, thick=0.0172, couple=0.85),
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def finger_rest_frame(name):
    """3x3 rest rotation (columns: lateral X, forward Y, dorsal Z) of the finger's base joint in hand space."""
    f = FINGERS[name]
    if name == "thumb":
        a = Vector(f["dir"]).normalized()
        n = Vector(f["nail"])
        n = (n - a * n.dot(a)).normalized()
    else:
        a = (Matrix.Rotation(-f["splay"], 3, "Z") @ Vector((0, 1, 0))).normalized()
        n = Vector((0, 0, 1))
    l = a.cross(n).normalized()
    m = Matrix((l, a, n)).transposed()  # columns
    return m


def pad_point(name):
    """Contact pad centre in the distal segment's local frame."""
    f = FINGERS[name]
    L = f["lens"][2]
    t = f["thick"] * 0.86
    return Vector((0.0, 0.66 * L, -0.56 * t))


CAM_END_POS = Vector((0.62, -1.62, 1.20))  # the robot looks into this lens at the end

MIRROR = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))

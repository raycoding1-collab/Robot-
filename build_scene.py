"""Build the complete scene procedurally and save robot_rubiks.blend.

Usage (Blender's Python, e.g. `pip install bpy kociemba` or a Blender install):
    python build_scene.py [--preset draft|preview|final|cinema4k] [--gpu] [--out robot_rubiks.blend]
    blender -b -P build_scene.py -- --preset final
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402

from robot_rubiks import animate, cameras, cube_model, details, environment, render_setup, robot_model  # noqa: E402
from robot_rubiks.choreography import Choreography  # noqa: E402
from robot_rubiks.rig_def import EYE_OFFSET, FPS  # noqa: E402


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="preview")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "robot_rubiks.blend"))
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args(argv)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def coll(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def main():
    args = parse()
    t0 = time.time()
    reset()
    c_env, c_robot, c_cube, c_cam = coll("Environment"), coll("Robot"), coll("PuzzleCube"), coll("Cameras")
    environment.build_environment(c_env)
    environment.setup_world()
    environment.add_bounce(c_env)
    mats = robot_model.Mats()
    J, parts = robot_model.build_robot(c_robot, mats)
    details.add_details(parts, mats, c_robot)
    cube_root, cubies = cube_model.build_cube(c_cube)
    choreo = Choreography(seed=args.seed)
    print("scramble :", " ".join(map(str, choreo.scramble)))
    print("solution :", " ".join(map(str, choreo.solution)), f"({len(choreo.solution)} moves)")
    print(f"solve    : {choreo.solve.t_start:.2f}s -> {choreo.solve.t_last_settle:.2f}s "
          f"({choreo.solve.t_last_settle - choreo.solve.t_start:.2f}s)")
    rig, frames = animate.bake(choreo, J, cube_root, cubies, eye_mat=mats.eye)

    def head_world(t):
        pose = rig.evaluate(t)
        m = J["root"].matrix_world.copy()
        from robot_rubiks.rig_def import PELVIS_POS
        from mathutils import Matrix
        m = Matrix.Translation(PELVIS_POS)
        for name in ("spine", "chest", "neck", "head"):
            m = m @ pose[name]
        return m @ EYE_OFFSET

    cameras.build_shots(choreo, c_cam, head_world)
    n_frames = int(choreo.duration * FPS)
    render_setup.configure(args.preset, frame_end=n_frames, use_gpu=args.gpu)
    sc = bpy.context.scene
    sc.frame_set(1)
    sc["scramble"] = " ".join(map(str, choreo.scramble))
    sc["solution"] = " ".join(map(str, choreo.solution))
    bpy.ops.wm.save_as_mainfile(filepath=args.out, compress=True)
    print(f"saved {args.out} ({n_frames} frames @ {FPS} fps) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

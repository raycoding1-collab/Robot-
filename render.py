"""Render frames / stills from robot_rubiks.blend and encode a video.

    python render.py --stills 40,120,300         # selected frames -> renders/stills
    python render.py --anim [--start 1 --end 522] # PNG sequence -> renders/frames
    python render.py --encode                     # frames -> renders/robot_rubiks.mp4
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def args_():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", default=os.path.join(HERE, "robot_rubiks.blend"))
    ap.add_argument("--stills", default="")
    ap.add_argument("--anim", action="store_true")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--pct", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "renders"))
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    return ap.parse_args(argv)


def render(a):
    import bpy

    sys.path.insert(0, HERE)
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    sc = bpy.context.scene
    if a.preset:
        from robot_rubiks import render_setup

        render_setup.configure(a.preset, use_gpu=a.gpu)
    if a.samples:
        sc.cycles.samples = a.samples
    if a.pct:
        sc.render.resolution_percentage = a.pct
    if a.threads:
        sc.render.threads_mode = "FIXED"
        sc.render.threads = a.threads
    frames = []
    if a.stills:
        frames = [int(x) for x in a.stills.split(",") if x]
        outdir = os.path.join(a.out, "stills")
    else:
        s = a.start or sc.frame_start
        e = a.end or sc.frame_end
        frames = list(range(s, e + 1, a.step))
        outdir = os.path.join(a.out, "frames")
    os.makedirs(outdir, exist_ok=True)
    for f in frames:
        path = os.path.join(outdir, f"f{f:04d}.png")
        if a.anim and os.path.exists(path):
            continue
        sc.frame_set(f)
        sc.render.filepath = path
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"frame {f} -> {path} ({time.time() - t:.1f}s)", flush=True)


def encode(a):
    frames = os.path.join(a.out, "frames", "f%04d.png")
    out = os.path.join(a.out, "robot_rubiks.mp4")
    vf = "vignette=angle=PI/5.2:mode=forward,noise=alls=3:allf=t,format=yuv420p"
    cmd = ["ffmpeg", "-y", "-framerate", "24", "-i", frames]
    audio = os.path.join(a.out, "foley.wav")
    if os.path.exists(audio):
        cmd += ["-i", audio, "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-movflags", "+faststart", out]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    a = args_()
    if a.encode:
        encode(a)
    else:
        render(a)

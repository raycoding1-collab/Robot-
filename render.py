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
    ap.add_argument("--target-mb", type=float, default=None, help="with --encode: two-pass encode that fits this many MB (10^6 bytes)")
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


def _check_frames(a, frames):
    # ffmpeg stops reading an image sequence at the first gap and would cut the audio
    # to match, so refuse to encode unless every frame is present and complete.
    sys.path.insert(0, os.path.join(HERE, "tools"))
    from render_watchdog import png_complete

    fdir = os.path.join(a.out, "frames")
    nums = sorted(int(f[1:5]) for f in os.listdir(fdir) if f.startswith("f") and f.endswith(".png"))
    first, last = a.start or 1, a.end or (nums[-1] if nums else 0)
    bad = [n for n in range(first, last + 1) if not png_complete(frames % n)]
    if not nums or bad:
        sys.exit(f"encode: {len(bad)} frames missing or incomplete in {first}-{last}: {bad[:20]}")
    return first, last


def encode(a):
    frames = os.path.join(a.out, "frames", "f%04d.png")
    first, last = _check_frames(a, frames)
    fps = 24
    # H.264 4:2:0 needs even dimensions: the 66 % preview preset gives 1267x712, so drop the
    # odd pixel column (lossless, unlike rescaling). Convert with the BT.709 matrix and tag it,
    # so players do not shift the colours.
    vf = ("crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0,vignette=angle=PI/5.2:mode=forward,noise=alls=3:allf=t,"
          "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p")
    color = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    video_in = ["-framerate", str(fps)] + (["-start_number", str(first)] if first != 1 else []) + ["-i", frames]
    audio = os.path.join(a.out, "foley.wav")
    has_audio = os.path.exists(audio)
    inputs = video_in + (["-i", audio] if has_audio else [])
    a_kbps = 192
    acodec = ["-c:a", "aac", "-b:a", f"{a_kbps}k", "-ar", "48000", "-shortest"] if has_audio else []
    x264 = ["-c:v", "libx264", "-profile:v", "high", "-level:v", "4.1"]

    if not a.target_mb:
        out = os.path.join(a.out, "robot_rubiks.mp4")
        cmd = ["ffmpeg", "-y", *inputs, "-vf", vf, *x264, "-preset", "slow", "-crf", "16", *color, *acodec,
               "-movflags", "+faststart", out]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    # Size-capped version (e.g. an Instagram DM, 25 MB): two-pass x264 at the largest bitrate that
    # still fits, then measure the file and correct the bitrate until it lands just under the cap.
    limit = int(a.target_mb * 1_000_000)  # decimal MB: the stricter reading of "25 MB"
    duration = (last - first + 1) / fps
    out = os.path.join(a.out, f"robot_rubiks_{a.target_mb:g}MB.mp4")
    passlog = os.path.join(a.out, "x264_2pass")
    kbps = (limit * 0.985 * 8 / duration) / 1000 - (a_kbps if has_audio else 0)
    slow = ["-preset", "veryslow", "-tune", "film"]
    subprocess.run(["ffmpeg", "-y", *video_in, "-vf", vf, *x264, *slow, "-b:v", f"{kbps:.0f}k",
                    "-pass", "1", "-passlogfile", passlog, "-an", "-f", "null", os.devnull], check=True)
    # Size is monotonic in bitrate but not proportional, so bracket it: keep the largest encode
    # that fits and interpolate between the best fitting and the smallest too-large attempt.
    goal, band = 0.99 * limit, (0.985 * limit, 0.996 * limit)
    part = out[:-4] + ".part.mp4"
    if os.path.exists(out):
        os.remove(out)
    fit = over = None  # (kbps, bytes)
    for attempt in range(8):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-vf", vf, *x264, *slow, "-b:v", f"{kbps:.0f}k",
                        "-pass", "2", "-passlogfile", passlog, *color, *acodec, "-movflags", "+faststart", part], check=True)
        size = os.path.getsize(part)
        print(f"attempt {attempt + 1}: {kbps:.0f} kb/s video -> {size} B ({size / limit:.2%} of {limit} B)", flush=True)
        if size <= band[1]:
            if fit is None or size > fit[1]:
                fit = (kbps, size)
                os.replace(part, out)
            if size >= band[0]:
                break
        elif over is None or size < over[1]:
            over = (kbps, size)
        if fit and over:
            kbps = fit[0] + (goal - fit[1]) * (over[0] - fit[0]) / (over[1] - fit[1])
        else:
            kbps *= goal / size
    for f in os.listdir(a.out):
        if f.startswith("x264_2pass") or f.endswith(".part.mp4"):
            os.remove(os.path.join(a.out, f))
    if fit is None:
        sys.exit(f"encode: could not fit {out} under {limit} bytes")
    print(f"wrote {out}: {fit[1]} B, {fit[1] / limit:.2%} of the cap", flush=True)


if __name__ == "__main__":
    a = args_()
    if a.encode:
        encode(a)
    else:
        render(a)

"""Generate the foley track (renders/foley.wav) and a verification report.

    python make_sound.py [--out renders] [--seed 7]
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: F401,E402  (mathutils)
import numpy as np  # noqa: E402

from robot_rubiks import animate, cameras, sound  # noqa: E402
from robot_rubiks.choreography import Choreography  # noqa: E402
from robot_rubiks.rig_def import FPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "renders"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    os.makedirs(a.out, exist_ok=True)
    c = Choreography(seed=a.seed)
    shots = cameras.shot_list(c)
    rig = animate.Rig({}, c)
    fol = sound.Foley(c, shots)
    raw = fol.render(rig)
    x = sound.Foley.master(raw)
    # stems at the same master gain, for balance verification
    g = 10 ** ((-23.0 - sound.integrated_lufs(raw)) / 20)
    np.savez_compressed(os.path.join(a.out, "foley_stems.npz"), **{k: (v * g).astype(np.float32) for k, v in fol.mix.stems.items()})
    path = os.path.join(a.out, "foley.wav")
    sound.write_wav(path, x)
    events = sorted(fol.events)
    with open(os.path.join(a.out, "foley_events.json"), "w") as f:
        json.dump([dict(t=round(t, 5), kind=k, strength=round(float(s), 3)) for t, k, s in events], f, indent=1)
    n_frames = int(c.duration * FPS)
    print(f"wrote {path}: {x.shape[1] / sound.SR:.4f}s (video {n_frames / FPS:.4f}s), {len(events)} events")
    print(f"integrated loudness {sound.integrated_lufs(x):.2f} LUFS, sample peak {20 * math.log10(np.max(np.abs(x))):.2f} dBFS")


if __name__ == "__main__":
    main()

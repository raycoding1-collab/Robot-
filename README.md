# AX-07 — a humanoid robot solves a speed cube

A fully procedural, physically driven Blender (Cycles) short: a modern humanoid
robot sits at a lab desk in late-afternoon daylight, picks up a scrambled 3×3
speed cube, studies every face carefully, then solves it at speed with finger
tricks, presents it and sets it down. It finishes by looking into the lens.

Everything is generated from Python: geometry, materials, rig, animation,
lighting, cameras and the sound. There are no external assets.

```
build_scene.py      -> builds robot_rubiks.blend (scene + baked animation + cameras)
render.py           -> renders stills / the frame sequence, encodes the MP4 (+ sound)
make_sound.py       -> synthesises renders/foley.wav from the same simulation
tools/verify_sound.py -> objective audio QA (sync, loudness, artefacts, balance)
tools/tune_grips.py -> optimiser that produced the hand grips used in the solve
tools/render_watchdog.py -> keeps a long multi-worker render alive (restarts crashed or stalled workers)
robot_rubiks/       -> the library (see below)
```

## What is simulated (and how)

| Aspect | Implementation |
| --- | --- |
| Cube state | Exact cubie model (`cube_logic.py`). The scramble `F D2 U L F2 U2 R U' D R L' U2 R2 B2 U2 L' U L F' R2 U2 F2` is solved by Kociemba's two-phase algorithm in 20 moves: `D' F R' D2 L' F2 D R' B' U2 R U D L2 B2 L2 U' B2 R2 D'`. That is 28 quarter turns in about 4.2 s. The solution is verified by simulation before anything is animated. |
| Layer dynamics | Finger-flicked turns are integrated physically (`physics.py`): the fingertip accelerates the layer to about 40°, releases it, and the layer coasts against Coulomb and viscous friction until the magnetic detents capture it. The detents produce a real overshoot of about 3° and a damped settle. Wrist turns are hand-driven along minimum-jerk profiles. |
| Reaction | Every layer acceleration applies an equal and opposite torque to the compliant grip. The whole cube and both hands recoil by fractions of a degree (a spring-damper model). |
| Contacts | Every fingertip that touches the cube follows the exact moving facelet of the cubie it is pushing. Return strokes follow lifted paths around the cube's square cross-section, so fingers never pass through plastic. |
| Robot | Rigid hard-surface parts on joint empties, with exact motion blur and no skinning. Arms use analytic two-bone IK with forearm roll. Fingers use a coupled three-link planar IK (DIP ≈ 0.72 × PIP). Grip frames were optimised numerically for reach, no penetration of the cube or desk, and natural joint ranges (`tools/tune_grips.py`). |
| Technique | U/U' are right/left index flicks; D/D' are left/right ring flicks; R/L are wrist turns (the palm does 78 %, the fingers the rest); F/B are thumb and middle-finger flicks. A resource scheduler keeps the cube held by the other fingers during every move and handles regrips. |
| Secondary motion | Critically damped head tracking with saccades during inspection, torso lean when reaching, servo micro-jitter and a breathing-like sway. |
| Look | Physical sky and sun through a glazed corner, linear LED fixtures, and a soft bounce light. Materials carry per-part colour variation, crevice dirt, smudges and fingerprints, micro-scratches and orange-peel clear coat. Fasteners and printed markings are ray-cast onto the shells. |
| Camera | Seven physical cameras (32–100 mm) with real apertures (7-blade bokeh), focus pulls, operator handheld drift, a 180° shutter, fog-glow bloom, lateral chromatic aberration, vignette and grain. |

## Sound

`make_sound.py` renders 48 kHz / 24-bit stereo foley from the same simulation the picture uses:

- Each layer turn produces a plastic slide whose level follows the layer's angular speed, then a modal-synthesis clack at the exact instant the magnets capture it.
- Fingertip taps, the cube lifting off and landing on the desk, and the hands settling on the desk.
- Servo and gear whir for six joint groups, with pitch and level driven by the rig's real joint velocities.
- A power-on relay, a cooling fan that spins up while the robot thinks, and room tone (HVAC plus a distant car through the glass).
- The microphone follows the edit: distance law and stereo position follow the active camera, with a convolution room reverb (RT60 ≈ 0.55 s).

It is mastered to EBU R128 (−23 LUFS, true peak ≤ −1 dBTP). `tools/verify_sound.py` checks 21 properties:

- duration matched to the picture
- no clipping, DC offset, pops or steady whine
- loudness, confirmed independently with ffmpeg
- all 28 clicks detected within 3 ms of the physics, each at the frame where the layer reaches 90°
- no unexplained transients
- mix balance per stem

The latest report is in `renders/foley_report.txt`.

## Running it

Requirements: Blender 5.x (or `pip install bpy==5.3.0`, Python 3.13), plus
`pip install kociemba numpy scipy` (and `matplotlib` for the QA plots).

```bash
python build_scene.py --preset final          # writes robot_rubiks.blend
python make_sound.py                           # writes renders/foley.wav
python tools/verify_sound.py                   # audio QA report + plots
python render.py --anim --preset final --gpu   # renders/frames/f0001.png ...
python tools/render_watchdog.py --workers 2    # or: CPU workers that restart themselves after a crash
python render.py --encode                      # renders/robot_rubiks.mp4 (with sound)
```

Inside Blender, open `robot_rubiks.blend`. The cuts are timeline markers bound to cameras.

Presets (`robot_rubiks/render_setup.py`):

| preset | resolution | samples | notes |
| --- | --- | --- | --- |
| draft | 960×540 | 16 | fast GI |
| preview | 66 % (1267×712; the MP4 is cropped to 1266×712) | 20 (adaptive) | fast GI, reduced bounces. Used for the CPU render in `renders/`. |
| final | 1920×1080 | 256 | full GI. GPU recommended (OptiX/CUDA/HIP/Metal auto-detected with `--gpu`). |
| cinema4k | 3840×2160 | 512 | |

The render in `renders/` was produced on a 4-core CPU with no GPU, so it uses the
`preview` preset. For the best image, render `final` on a GPU.

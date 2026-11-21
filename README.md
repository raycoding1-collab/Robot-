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

`make_sound.py` renders 48 kHz / 24-bit stereo foley from the same simulation the picture uses. One rule applies throughout: there are no steady pure tones in the mix (no motor whine, fan blade tone or mains hum).

- **Layer turns.** While a layer slides, it makes a soft, dark stick-slip scrape (lubricated plastic): distinct slips whose rate and strength follow the layer's angular speed. When the magnets capture it, 2–4 of its cubies seat against their neighbours within about 3 ms. Each seating is a Hertzian contact (0.32–0.45 ms) exciting hand-damped shell modes (Q ≈ 10–20), plus the low "thock" of the cube body, so the result is a short plastic clack, not a ringing ping. A 1.5 ms transient shaper softens only the first instant of each contact.
- **Robot motors.** A faint, pitch-free mechanical whirr. It is silent while a joint holds or creeps, fades in only for real movements (arms above 1.5 rad/s, fingers above 3 rad/s), and builds up and dies away with rotor inertia.
- **Handling.** Soft rubber fingertip taps, an almost inaudible grip creak (the cube turns rigidly in the hands), the cube lifting off and being set down on the desk, and the hands settling. Lifting the hands off the desk makes no sound.
- **Electronics and room.** A tiny power-on relay, a cooling fan heard only as faint airflow, and a quiet room tone (HVAC plus a distant car through the glass).
- **Perspective.** The microphone follows the edit with a film-style perspective (gentler than the 1/d law, so cuts don't jump in level) and a convolution room reverb (RT60 ≈ 0.55 s).

The mix keeps the film's quiet moments quiet. The robot inspecting the cube is near-silent, as it would be in a real quiet room. The master uses EBU R128 normalisation (−23 LUFS) and a true-peak limiter.

`tools/verify_sound.py` checks 27 properties:

- **Format and artefacts:** duration matched to the picture; no clipping, DC offset or pops.
- **Loudness:** confirmed independently with ffmpeg.
- **Sync:** all 28 clicks detected within 3 ms of the physics, each at the frame where the layer reaches 90°. No unexplained transients.
- **Mix balance:** checked per stem.
- **Listening comfort and realism:**
  - no steady tone anywhere in the film (scanned in 3 s windows)
  - nothing outside the solve louder than −10 LU against the film's loudness
  - robot motors at most −16 LU at their loudest
  - clicks are damped contacts (ring-down ≤ 25 ms, resonance Q ≤ 30)
  - the turning sound is grainy friction, not hiss
  - no pitch region dominates the solve
  - no extreme loudness jump into the solve

The two earlier sound versions fail these checks:

- **First version:** ringing clicks (Q 88), a 2 kHz pile-up, hiss, and a 16 dB steady tone.
- **Second version:** motor whine through the whole film (a 14 dB tone), and handling sounds as loud as the film itself.

The latest report is in `renders/foley_report.txt`. `python render.py --reaudio` swaps a new `foley.wav` into the finished MP4s without re-encoding the picture.

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
python render.py --encode --target-mb 25      # size-capped copy (two-pass, lands just under 25 MB)
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

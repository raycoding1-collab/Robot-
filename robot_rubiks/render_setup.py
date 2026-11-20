"""Render / colour-management / compositor configuration."""
from __future__ import annotations

import bpy

PRESETS = {
    # name: (resolution %, samples, adaptive threshold, fast GI / reduced bounces)
    "draft": (50, 16, 0.05, True),
    "preview": (66, 20, 0.035, True),  # 1280x720, CPU friendly (~30-45 s/frame on 4 cores)
    "final": (100, 256, 0.012, False),  # 1920x1080, full GI (GPU recommended)
    "cinema4k": (200, 512, 0.01, False),
}


def configure(preset="preview", frame_end=None, use_gpu=False):
    sc = bpy.context.scene
    pct, samples, thr, fast = PRESETS[preset]
    r = sc.render
    r.engine = "CYCLES"
    r.resolution_x = 1920
    r.resolution_y = 1080
    r.resolution_percentage = pct
    r.fps = 24
    r.fps_base = 1.0
    r.film_transparent = False
    r.use_motion_blur = True
    r.motion_blur_shutter = 0.5  # 180 degree shutter
    r.motion_blur_position = "CENTER"
    r.use_persistent_data = True
    cy = sc.cycles
    cy.device = "GPU" if use_gpu else "CPU"
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = thr
    cy.adaptive_min_samples = 0
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.denoising_prefilter = "ACCURATE"
    try:
        cy.denoising_quality = "HIGH"
    except Exception:
        pass
    cy.use_light_tree = True
    cy.max_bounces = 10
    cy.diffuse_bounces = 4
    cy.glossy_bounces = 4
    cy.transmission_bounces = 8
    cy.transparent_max_bounces = 12
    cy.volume_bounces = 0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.blur_glossy = 0.5
    if fast:
        cy.diffuse_bounces = 2
        cy.glossy_bounces = 3
        cy.transmission_bounces = 6
        cy.max_bounces = 8
        cy.use_fast_gi = True
        cy.fast_gi_method = "REPLACE"
        cy.ao_bounces_render = 1
        if sc.world:
            sc.world.light_settings.distance = 3.0
    else:
        cy.use_fast_gi = False
    cy.sample_clamp_direct = 0.0
    cy.sample_clamp_indirect = 6.0
    try:
        cy.film_exposure = 1.0
    except Exception:
        pass
    vs = sc.view_settings
    vs.view_transform = "AgX"
    for look in ("AgX - Medium High Contrast", "AgX - Base Contrast", "None"):
        try:
            vs.look = look
            break
        except Exception:
            continue
    vs.exposure = 0.0
    vs.gamma = 1.0
    r.image_settings.file_format = "PNG"
    r.image_settings.color_depth = "8"
    if frame_end is not None:
        sc.frame_start = 1
        sc.frame_end = frame_end
    if use_gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                if any(d.type == backend for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = True
                    break
            except Exception:
                continue
    setup_compositor()


def setup_compositor():
    """Subtle optical bloom (fog glow) + lateral chromatic aberration of a real lens."""
    sc = bpy.context.scene
    try:
        ng = bpy.data.node_groups.new("LensComp", "CompositorNodeTree")
        rl = ng.nodes.new("CompositorNodeRLayers")
        glare = ng.nodes.new("CompositorNodeGlare")
        try:
            glare.inputs["Type"].default_value = "Fog Glow"
        except Exception:
            try:
                glare.glare_type = "FOG_GLOW"
            except Exception:
                pass
        for k, v in (("Threshold", 1.2), ("Strength", 0.12), ("Size", 0.5), ("Quality", "High")):
            try:
                glare.inputs[k].default_value = v
            except Exception:
                pass
        lens = ng.nodes.new("CompositorNodeLensdist")
        for k, v in (("Distortion", -0.004), ("Dispersion", 0.006)):
            try:
                lens.inputs[k].default_value = v
            except Exception:
                pass
        out = ng.nodes.new("NodeGroupOutput")
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        ng.links.new(rl.outputs["Image"], glare.inputs["Image"])
        ng.links.new(glare.outputs["Image"], lens.inputs["Image"])
        ng.links.new(lens.outputs["Image"], out.inputs[0])
        sc.compositing_node_group = ng
        sc.render.use_compositing = True
    except Exception as e:  # compositor is optional polish
        print("compositor setup skipped:", e)

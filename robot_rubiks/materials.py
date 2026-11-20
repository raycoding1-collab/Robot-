"""Physically based materials (Cycles). All procedural, no texture files needed."""
from __future__ import annotations

import bpy


def srgb_to_lin(c):
    def f(x):
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    return tuple(f(x) for x in c)


class NodeBuilder:
    def __init__(self, mat):
        self.mat = mat
        self.nt = mat.node_tree
        self.nodes = self.nt.nodes
        self.links = self.nt.links
        self.x = -900

    def node(self, kind, **inputs):
        n = self.nodes.new(kind)
        n.location = (self.x, len(self.nodes) * -40)
        for k, v in inputs.items():
            if k.startswith("_"):
                setattr(n, k[1:], v)
            else:
                n.inputs[k].default_value = v
        return n

    def link(self, a, b):
        self.links.new(a, b)


def new_material(name):
    m = bpy.data.materials.get(name)
    if m:
        return m, None
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nb = NodeBuilder(m)
    return m, nb


def principled(nb):
    return nb.nodes.get("Principled BSDF")


def _micro_noise(nb, scale, detail=6.0):
    tc = nb.node("ShaderNodeTexCoord")
    nz = nb.node("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=0.55)
    nb.link(tc.outputs["Object"], nz.inputs["Vector"])
    return nz


def _rough_variation(nb, bsdf, lo, hi, scale=60.0):
    nz = _micro_noise(nb, scale, 4.0)
    mr = nb.node("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = lo
    mr.inputs["To Max"].default_value = hi
    mr.inputs["From Min"].default_value = 0.3
    mr.inputs["From Max"].default_value = 0.7
    nb.link(nz.outputs["Fac"], mr.inputs["Value"])
    nb.link(mr.outputs["Result"], bsdf.inputs["Roughness"])
    return nz


def _bump(nb, bsdf, scale, strength, distance=0.0005, coat=False):
    nz = _micro_noise(nb, scale, 8.0)
    bp = nb.node("ShaderNodeBump", Strength=strength, Distance=distance)
    nb.link(nz.outputs["Fac"], bp.inputs["Height"])
    nb.link(bp.outputs["Normal"], bsdf.inputs["Normal"])
    if coat:
        nb.link(bp.outputs["Normal"], bsdf.inputs["Coat Normal"])
    return bp


def _imperfections(nb, bsdf, base_rgb, rough_lo, rough_hi, coat=0.0, peel=0.05, smudge=1.0, scratch=1.0, dirt=0.22, tint_var=0.025):
    """Layer real-world wear onto a Principled BSDF:
    per-part colour variation, crevice dirt (AO), smudges/fingerprints and
    micro-scratches in roughness, orange-peel waviness in the clear coat."""
    tc = nb.node("ShaderNodeTexCoord")
    oi = nb.node("ShaderNodeObjectInfo")
    # --- per-part tint variation (moulded parts never match perfectly)
    var = nb.node("ShaderNodeMapRange")
    var.inputs["To Min"].default_value = 1.0 - tint_var
    var.inputs["To Max"].default_value = 1.0 + tint_var * 0.6
    nb.link(oi.outputs["Random"], var.inputs["Value"])
    base = nb.node("ShaderNodeRGB")
    base.outputs[0].default_value = (*base_rgb, 1)
    mul = nb.node("ShaderNodeMix", _data_type="RGBA", _blend_type="MULTIPLY")
    mul.inputs["Factor"].default_value = 1.0
    nb.link(base.outputs[0], mul.inputs[6])
    nb.link(var.outputs["Result"], mul.inputs[7])
    col = mul.outputs[2]
    # --- crevice dirt / dust from local ambient occlusion
    if dirt > 0:
        ao = nb.node("ShaderNodeAmbientOcclusion", _samples=6, _only_local=True)
        ao.inputs["Distance"].default_value = 0.012
        dm = nb.node("ShaderNodeMix", _data_type="RGBA", _blend_type="MIX")
        grime = nb.node("ShaderNodeRGB")
        grime.outputs[0].default_value = (base_rgb[0] * 0.42, base_rgb[1] * 0.40, base_rgb[2] * 0.37, 1)
        inv = nb.node("ShaderNodeMath", _operation="SUBTRACT")
        inv.inputs[0].default_value = 1.0
        nb.link(ao.outputs["AO"], inv.inputs[1])
        fac = nb.node("ShaderNodeMath", _operation="MULTIPLY")
        fac.inputs[1].default_value = dirt
        nb.link(inv.outputs[0], fac.inputs[0])
        nb.link(fac.outputs[0], dm.inputs["Factor"])
        nb.link(col, dm.inputs[6])
        nb.link(grime.outputs[0], dm.inputs[7])
        col = dm.outputs[2]
    nb.link(col, bsdf.inputs["Base Color"])
    # --- roughness: smudges + fingerprints + micro scratches
    sm = nb.node("ShaderNodeTexNoise", Scale=7.0, Detail=9.0, Roughness=0.62, Distortion=1.8)
    nb.link(tc.outputs["Object"], sm.inputs["Vector"])
    smr = nb.node("ShaderNodeMapRange")
    smr.inputs["From Min"].default_value = 0.45
    smr.inputs["From Max"].default_value = 0.72
    smr.inputs["To Min"].default_value = 0.0
    smr.inputs["To Max"].default_value = smudge
    nb.link(sm.outputs["Fac"], smr.inputs["Value"])
    # fingerprint whorls: distorted rings, masked to a few patches
    fp_map = nb.node("ShaderNodeMapping")
    fp_map.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    nb.link(tc.outputs["Object"], fp_map.inputs["Vector"])
    fp = nb.node("ShaderNodeTexWave", Scale=420.0, Distortion=9.0, Detail=2.0)
    fp.wave_type = "RINGS"
    nb.link(fp_map.outputs["Vector"], fp.inputs["Vector"])
    fpm = nb.node("ShaderNodeTexNoise", Scale=30.0, Detail=2.0)
    nb.link(tc.outputs["Object"], fpm.inputs["Vector"])
    fpm_r = nb.node("ShaderNodeMapRange")
    fpm_r.inputs["From Min"].default_value = 0.62
    fpm_r.inputs["From Max"].default_value = 0.70
    nb.link(fpm.outputs["Fac"], fpm_r.inputs["Value"])
    fpx = nb.node("ShaderNodeMath", _operation="MULTIPLY")
    nb.link(fp.outputs["Fac"], fpx.inputs[0])
    nb.link(fpm_r.outputs["Result"], fpx.inputs[1])
    # scratches: very stretched noise, thresholded into thin lines
    sc_map = nb.node("ShaderNodeMapping")
    sc_map.inputs["Scale"].default_value = (900.0, 6.0, 900.0)
    sc_map.inputs["Rotation"].default_value = (0.4, 0.9, 0.2)
    nb.link(tc.outputs["Object"], sc_map.inputs["Vector"])
    scn = nb.node("ShaderNodeTexNoise", Scale=1.0, Detail=1.0)
    nb.link(sc_map.outputs["Vector"], scn.inputs["Vector"])
    scr = nb.node("ShaderNodeMapRange")
    scr.inputs["From Min"].default_value = 0.70
    scr.inputs["From Max"].default_value = 0.78
    scr.inputs["To Max"].default_value = scratch
    nb.link(scn.outputs["Fac"], scr.inputs["Value"])
    add1 = nb.node("ShaderNodeMath", _operation="ADD")
    nb.link(smr.outputs["Result"], add1.inputs[0])
    nb.link(fpx.outputs[0], add1.inputs[1])
    add2 = nb.node("ShaderNodeMath", _operation="ADD", _use_clamp=True)
    nb.link(add1.outputs[0], add2.inputs[0])
    nb.link(scr.outputs["Result"], add2.inputs[1])
    rr = nb.node("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value = rough_lo
    rr.inputs["To Max"].default_value = rough_hi
    nb.link(add2.outputs[0], rr.inputs["Value"])
    nb.link(rr.outputs["Result"], bsdf.inputs["Roughness"])
    if coat > 0:
        bsdf.inputs["Coat Weight"].default_value = coat
        crr = nb.node("ShaderNodeMapRange")
        crr.inputs["To Min"].default_value = 0.035
        crr.inputs["To Max"].default_value = 0.22
        nb.link(add2.outputs[0], crr.inputs["Value"])
        nb.link(crr.outputs["Result"], bsdf.inputs["Coat Roughness"])
        # orange peel of sprayed clear coat (a few mm wavelength)
        op = nb.node("ShaderNodeTexNoise", Scale=260.0, Detail=3.0, Roughness=0.5)
        nb.link(tc.outputs["Object"], op.inputs["Vector"])
        bp = nb.node("ShaderNodeBump", Strength=peel, Distance=0.0003)
        nb.link(op.outputs["Fac"], bp.inputs["Height"])
        nb.link(bp.outputs["Normal"], bsdf.inputs["Coat Normal"])
    # base micro texture + scratch grooves
    mt = nb.node("ShaderNodeTexNoise", Scale=1800.0, Detail=3.0)
    nb.link(tc.outputs["Object"], mt.inputs["Vector"])
    hs = nb.node("ShaderNodeMath", _operation="ADD")
    nb.link(mt.outputs["Fac"], hs.inputs[0])
    sg = nb.node("ShaderNodeMath", _operation="MULTIPLY")
    sg.inputs[1].default_value = -0.6
    nb.link(scr.outputs["Result"], sg.inputs[0])
    nb.link(sg.outputs[0], hs.inputs[1])
    bp2 = nb.node("ShaderNodeBump", Strength=0.05, Distance=0.0001)
    nb.link(hs.outputs[0], bp2.inputs["Height"])
    nb.link(bp2.outputs["Normal"], bsdf.inputs["Normal"])


def ceramic_white():
    m, nb = new_material("Robot_CeramicWhite")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Specular IOR Level"].default_value = 0.5
    b.inputs["Subsurface Weight"].default_value = 0.03
    b.inputs["Subsurface Radius"].default_value = (0.002, 0.002, 0.002)
    _imperfections(nb, b, srgb_to_lin((0.875, 0.878, 0.872)), 0.30, 0.52, coat=0.6, peel=0.06, smudge=0.55, scratch=0.5, dirt=0.24)
    return m


def satin_white():
    """Matte injection-moulded polymer used for secondary panels."""
    m, nb = new_material("Robot_SatinWhite")
    if nb is None:
        return m
    b = principled(nb)
    _imperfections(nb, b, srgb_to_lin((0.80, 0.805, 0.80)), 0.50, 0.68, coat=0.0, smudge=0.5, scratch=0.4, dirt=0.28)
    return m


def graphite_metal():
    m, nb = new_material("Robot_Graphite")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Anisotropic"].default_value = 0.25
    _imperfections(nb, b, srgb_to_lin((0.22, 0.225, 0.235)), 0.30, 0.50, smudge=0.7, scratch=0.6, dirt=0.15, tint_var=0.04)
    return m


def dark_polymer():
    m, nb = new_material("Robot_DarkPolymer")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Base Color"].default_value = (*srgb_to_lin((0.045, 0.047, 0.05)), 1)
    b.inputs["Roughness"].default_value = 0.42
    b.inputs["Coat Weight"].default_value = 0.2
    b.inputs["Coat Roughness"].default_value = 0.2
    _bump(nb, b, 1800.0, 0.05, 0.0002)
    return m


def chrome():
    m, nb = new_material("Robot_Chrome")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Base Color"].default_value = (*srgb_to_lin((0.86, 0.865, 0.87)), 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = 0.12
    return m


def copper():
    m, nb = new_material("Robot_Copper")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Base Color"].default_value = (*srgb_to_lin((0.93, 0.62, 0.45)), 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = 0.28
    return m


def rubber():
    m, nb = new_material("Robot_Rubber")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Specular IOR Level"].default_value = 0.35
    b.inputs["Sheen Weight"].default_value = 0.2
    b.inputs["Sheen Roughness"].default_value = 0.4
    _imperfections(nb, b, srgb_to_lin((0.040, 0.041, 0.043)), 0.55, 0.78, smudge=0.6, scratch=0.2, dirt=0.0, tint_var=0.05)
    # lint / dust specks that black rubber always collects
    tc = nb.node("ShaderNodeTexCoord")
    vo = nb.node("ShaderNodeTexVoronoi")
    vo.inputs["Scale"].default_value = 900.0
    nb.link(tc.outputs["Object"], vo.inputs["Vector"])
    th = nb.node("ShaderNodeMath", _operation="LESS_THAN")
    th.inputs[1].default_value = 0.06
    nb.link(vo.outputs["Distance"], th.inputs[0])
    mix = nb.node("ShaderNodeMix", _data_type="RGBA")
    mix.inputs[7].default_value = (0.35, 0.34, 0.32, 1)
    link_from = b.inputs["Base Color"].links[0].from_socket
    nb.link(link_from, mix.inputs[6])
    nb.link(th.outputs[0], mix.inputs["Factor"])
    nb.link(mix.outputs[2], b.inputs["Base Color"])
    return m


def visor_glass():
    m, nb = new_material("Robot_Visor")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Base Color"].default_value = (0.20, 0.21, 0.22, 1)
    b.inputs["Roughness"].default_value = 0.012
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["IOR"].default_value = 1.52
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat Roughness"].default_value = 0.0
    return m


def face_inner():
    m, nb = new_material("Robot_FaceInner")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Base Color"].default_value = (0.004, 0.004, 0.005, 1)
    b.inputs["Roughness"].default_value = 0.35
    return m


def emitter(name, color, strength):
    m, nb = new_material(name)
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1)
    b.inputs["Roughness"].default_value = 0.2
    b.inputs["Emission Color"].default_value = (*color, 1)
    b.inputs["Emission Strength"].default_value = strength
    return m


def lens_glass():
    m, nb = new_material("Robot_Lens")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1)
    b.inputs["Roughness"].default_value = 0.02
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Thin Film Thickness"].default_value = 420.0
    b.inputs["Thin Film IOR"].default_value = 1.38
    return m


# ----------------------------------------------------------------------------
# Puzzle cube
# ----------------------------------------------------------------------------
CUBE_COLORS_SRGB = {
    "white": (0.93, 0.93, 0.92),
    "yellow": (1.00, 0.86, 0.02),
    "red": (0.80, 0.04, 0.07),
    "orange": (1.00, 0.40, 0.02),
    "blue": (0.00, 0.30, 0.86),
    "green": (0.02, 0.72, 0.28),
}


def cube_body():
    m, nb = new_material("Cube_Body")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Base Color"].default_value = (0.012, 0.012, 0.013, 1)
    b.inputs["Roughness"].default_value = 0.38
    _bump(nb, b, 3000.0, 0.08, 0.0001)
    return m


def cube_tile(color_name):
    m, nb = new_material(f"Cube_Tile_{color_name}")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Specular IOR Level"].default_value = 0.5
    b.inputs["Subsurface Weight"].default_value = 0.08
    b.inputs["Subsurface Radius"].default_value = (0.001, 0.001, 0.001)
    # frosted matte finish typical of stickerless speed cubes, handled = smudged
    _imperfections(nb, b, srgb_to_lin(CUBE_COLORS_SRGB[color_name]), 0.30, 0.46, coat=0.12, peel=0.02, smudge=0.6, scratch=0.5, dirt=0.0, tint_var=0.015)
    return m


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
def stone_table():
    m, nb = new_material("Env_Basalt")
    if nb is None:
        return m
    b = principled(nb)
    tc = nb.node("ShaderNodeTexCoord")
    n1 = nb.node("ShaderNodeTexNoise", Scale=8.0, Detail=10.0, Roughness=0.6)
    n2 = nb.node("ShaderNodeTexNoise", Scale=180.0, Detail=4.0, Roughness=0.5)
    nb.link(tc.outputs["Object"], n1.inputs["Vector"])
    nb.link(tc.outputs["Object"], n2.inputs["Vector"])
    ramp = nb.node("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0.018, 0.018, 0.019, 1)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (0.040, 0.039, 0.038, 1)
    mix = nb.node("ShaderNodeMix", _data_type="FLOAT")
    mix.inputs["Factor"].default_value = 0.35
    nb.link(n1.outputs["Fac"], mix.inputs[2])
    nb.link(n2.outputs["Fac"], mix.inputs[3])
    nb.link(mix.outputs[0], ramp.inputs["Fac"])
    nb.link(ramp.outputs["Color"], b.inputs["Base Color"])
    mr = nb.node("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 0.22
    mr.inputs["To Max"].default_value = 0.42
    nb.link(n1.outputs["Fac"], mr.inputs["Value"])
    nb.link(mr.outputs["Result"], b.inputs["Roughness"])
    bp = nb.node("ShaderNodeBump", Strength=0.05, Distance=0.0003)
    nb.link(n2.outputs["Fac"], bp.inputs["Height"])
    nb.link(bp.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Coat Weight"].default_value = 0.25
    b.inputs["Coat Roughness"].default_value = 0.08
    return m


def concrete_floor():
    m, nb = new_material("Env_Concrete")
    if nb is None:
        return m
    b = principled(nb)
    tc = nb.node("ShaderNodeTexCoord")
    n1 = nb.node("ShaderNodeTexNoise", Scale=1.2, Detail=12.0, Roughness=0.65)
    nb.link(tc.outputs["Object"], n1.inputs["Vector"])
    ramp = nb.node("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.035, 0.035, 0.036, 1)
    ramp.color_ramp.elements[1].color = (0.075, 0.073, 0.07, 1)
    nb.link(n1.outputs["Fac"], ramp.inputs["Fac"])
    nb.link(ramp.outputs["Color"], b.inputs["Base Color"])
    mr = nb.node("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 0.18
    mr.inputs["To Max"].default_value = 0.55
    nb.link(n1.outputs["Fac"], mr.inputs["Value"])
    nb.link(mr.outputs["Result"], b.inputs["Roughness"])
    return m


def wall_paint():
    m, nb = new_material("Env_Wall")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Base Color"].default_value = (0.028, 0.029, 0.031, 1)
    b.inputs["Roughness"].default_value = 0.85
    return m


def acoustic_panel():
    m, nb = new_material("Env_Panel")
    if nb is None:
        return m
    b = principled(nb)
    b.inputs["Base Color"].default_value = (0.05, 0.052, 0.056, 1)
    b.inputs["Roughness"].default_value = 0.6
    return m


def walnut():
    m, nb = new_material("Env_Walnut")
    if nb is None:
        return m
    b = principled(nb)
    tc = nb.node("ShaderNodeTexCoord")
    mp = nb.node("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 14.0, 1.0)
    nb.link(tc.outputs["Object"], mp.inputs["Vector"])
    wave = nb.node("ShaderNodeTexWave")
    wave.wave_type = "RINGS"
    wave.inputs["Scale"].default_value = 3.0
    wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 6.0
    nb.link(mp.outputs["Vector"], wave.inputs["Vector"])
    ramp = nb.node("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.035, 0.018, 0.010, 1)
    ramp.color_ramp.elements[1].color = (0.11, 0.060, 0.035, 1)
    nb.link(wave.outputs["Fac"], ramp.inputs["Fac"])
    nb.link(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.4
    b.inputs["Coat Weight"].default_value = 0.4
    b.inputs["Coat Roughness"].default_value = 0.15
    return m


def screen_emit(name, color, strength):
    m, nb = new_material(name)
    if nb is None:
        return m
    b = principled(nb)
    tc = nb.node("ShaderNodeTexCoord")
    nz = nb.node("ShaderNodeTexNoise", Scale=3.0, Detail=2.0)
    nb.link(tc.outputs["Generated"], nz.inputs["Vector"])
    ramp = nb.node("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (color[0] * 0.3, color[1] * 0.3, color[2] * 0.3, 1)
    ramp.color_ramp.elements[1].color = (*color, 1)
    nb.link(nz.outputs["Fac"], ramp.inputs["Fac"])
    nb.link(ramp.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = strength
    b.inputs["Base Color"].default_value = (0.01, 0.01, 0.01, 1)
    b.inputs["Roughness"].default_value = 0.1
    return m

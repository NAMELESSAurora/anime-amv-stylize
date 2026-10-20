# -*- coding: utf-8 -*-
"""
GPU 合成（moderngl，片元着色器逐像素）。用户 10-18："你可以调用 GPU 一起渲染，单用 CPU 占用太高了"。
profile（预览 B，8 帧）：CPU 每帧 ~0.93 s，大头是 HUD 预乘合成（0.33 s）、R 底（暗化 + 冷色网点 + 颗粒，0.47 s）、
C 暗态底（印刷灰阶 + 网点 + 颗粒 + 抠像混合，0.46 s）、PIL 图层合成（~0.3 s）——全是逐像素运算，挪到显卡。
算法和 CPU 版（v2_looks.r_base / c_dark、hud_v2.Canvas.over、v2_looks.compose）一致；颗粒用哈希噪声（同样 hold 2）。
坐标：texelFetch(ivec2(gl_FragCoord.xy)) + 直接读回，行序与 numpy 一致，不翻转。
每个进程一份 standalone context；没有 moderngl / 显卡时 ok() 返回 False，调用方退回 CPU。
"""
import math
import numpy as np

W, H = 2560, 1440
R = H / 1080.0
_S = {}

VS = """#version 330
in vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"""

HEAD = """#version 330
out vec4 o;
float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
float hash(vec2 p, float s){ return fract(sin(dot(p, vec2(127.1, 311.7)) + s * 74.7) * 43758.5453); }
float gauss(vec2 cell, float seed){
    float u1 = max(hash(cell, seed), 1e-4), u2 = hash(cell + 17.3, seed + 3.1);
    return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}
bool dot_on(vec2 xy, float g, float pitch, float ang){   // cards.halftone_c：点径随暗度
    float c = cos(ang), s = sin(ang);
    float u = mod(xy.x * c + xy.y * s, pitch) - pitch * 0.5;
    float v = mod(-xy.x * s + xy.y * c, pitch) - pitch * 0.5;
    return sqrt(u * u + v * v) / (pitch * 0.7071) < 1.0 - g;
}
"""

FS_OVER = HEAD + """
uniform sampler2D base; uniform sampler2D prem; uniform sampler2D txt; uniform int has_txt;
void main(){
    ivec2 c = ivec2(gl_FragCoord.xy);
    vec3 b = texelFetch(base, c, 0).rgb; vec4 p = texelFetch(prem, c, 0);
    vec3 r = b * (1.0 - p.a) + p.rgb;
    if (has_txt == 1) { vec4 t = texelFetch(txt, c, 0); r = r * (1.0 - t.a) + t.rgb * t.a; }
    o = vec4(r, 1.0);
}"""

FS_COMP = HEAD + """
uniform sampler2D base; uniform sampler2D lay;
void main(){
    ivec2 c = ivec2(gl_FragCoord.xy);
    vec3 b = texelFetch(base, c, 0).rgb; vec4 t = texelFetch(lay, c, 0);
    o = vec4(b * (1.0 - t.a) + t.rgb * t.a, 1.0);
}"""

FS_RBASE = HEAD + """
uniform sampler2D src; uniform float strength; uniform float seed; uniform float pitch; uniform float Rs;
void main(){
    ivec2 c = ivec2(gl_FragCoord.xy);
    vec3 s = texelFetch(src, c, 0).rgb * 255.0;
    float g = luma(s) / 255.0;
    float w = clamp((0.55 - g) / 0.35, 0.0, 1.0) * strength;
    vec3 r = s * (1.0 - 0.22 * strength);
    r = r * (1.0 - 0.28 * w) + vec3(18.0, 38.0, 50.0) * 0.28 * w;
    if (dot_on(vec2(c), g, pitch, radians(15.0))) r = r * (1.0 - 0.7 * w) + vec3(6.0, 12.0, 18.0) * 0.7 * w;
    r += gauss(floor(vec2(c) / 2.0), seed) * 5.0 * strength;
    o = vec4(clamp(r, 0.0, 255.0) / 255.0, 1.0);
}"""

FS_CDARK = HEAD + """
uniform sampler2D src; uniform sampler2D mask; uniform vec3 tint; uniform float seed; uniform float pitch;
void main(){
    ivec2 c = ivec2(gl_FragCoord.xy);
    vec3 s = texelFetch(src, c, 0).rgb * 255.0;
    float g = luma(s) / 255.0;
    vec3 tone = g * 150.0 * tint;
    vec3 bg = dot_on(vec2(c), g, pitch, radians(45.0)) ? tone * 0.35 : tone;
    bg += gauss(floor(vec2(c) / 2.0), seed) * 4.0;
    float m = texture(mask, (vec2(c) + 0.5) / vec2(%d.0, %d.0)).r;
    o = vec4(clamp(s * m + bg * (1.0 - m), 0.0, 255.0) / 255.0, 1.0);
}""" % (W, H)


def use():
    import os
    return os.environ.get("V2_GPU", "1") != "0" and ok()


def ok():
    if "ok" in _S:
        return _S["ok"]
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
        _S["ctx"] = ctx
        vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        for name, fs in (("over", FS_OVER), ("comp", FS_COMP), ("rbase", FS_RBASE), ("cdark", FS_CDARK)):
            prog = ctx.program(vertex_shader=VS, fragment_shader=fs)
            _S[name] = (prog, ctx.simple_vertex_array(prog, vbo, "p"))
        _S["fbo"] = ctx.simple_framebuffer((W, H), components=3)
        _S["tex"] = {}
        _S["ok"] = True
    except Exception as e:                                  # 没显卡 / 没驱动：退回 CPU
        print("gpu off:", e)
        _S["ok"] = False
    return _S["ok"]


def _tex(slot, arr, filt_linear=False):
    import moderngl
    ctx = _S["ctx"]
    h, w = arr.shape[:2]
    comp = 1 if arr.ndim == 2 else arr.shape[2]
    key = (slot, w, h, comp)
    t = _S["tex"].get(key)
    if t is None:
        t = ctx.texture((w, h), comp, dtype="f1")
        t.filter = (moderngl.LINEAR, moderngl.LINEAR) if filt_linear else (moderngl.NEAREST, moderngl.NEAREST)
        _S["tex"][key] = t
    t.write(np.ascontiguousarray(arr).tobytes())
    return t


def _run(name, texs, **uni):
    prog, vao = _S[name]
    for i, (uname, t) in enumerate(texs):
        t.use(i); prog[uname].value = i
    for k, v in uni.items():
        if k in prog:
            prog[k].value = v
    fbo = _S["fbo"]; fbo.use()
    vao.render(mode=_S["ctx"].TRIANGLE_STRIP)
    return np.frombuffer(fbo.read(components=3), np.uint8).reshape(H, W, 3)


def over(base, prem_rgba, txt_rgba=None):
    """HUD：预乘 RGBA（OpenCV 抗锯齿层）+ 直通 alpha 文字层 → base。"""
    texs = [("base", _tex(0, base)), ("prem", _tex(1, prem_rgba))]
    if txt_rgba is not None:
        texs.append(("txt", _tex(2, txt_rgba)))
    return _run("over", texs, has_txt=1 if txt_rgba is not None else 0)


def compose(base, lay_rgba):
    return _run("comp", [("base", _tex(0, base)), ("lay", _tex(3, lay_rgba))])


def r_base(src, f, strength=1.0):
    return _run("rbase", [("src", _tex(0, src))], strength=float(strength), seed=float(int(f) // 2 % 9973),
                pitch=float(7 * R), Rs=float(R))


def c_dark_base(src, mask_half, tint, seed):
    """mask_half：已模糊的抠像（任意分辨率 uint8，线性采样放大）。"""
    return _run("cdark", [("src", _tex(0, src)), ("mask", _tex(4, mask_half, filt_linear=True))],
                tint=tuple(float(x) for x in tint), seed=float(seed % 9973), pitch=float(7 * R))


# ---------------------------------------------------------------- HUD v3 整层合成（hud_v3.render3 的显卡版）
FS_HUD3 = HEAD + """
uniform sampler2D base; uniform sampler2D ba; uniform sampler2D bt; uniform sampler2D fa; uniform sampler2D ft;
uniform sampler2D fg; uniform sampler2D rows;
uniform mat3 binv; uniform float fade_b; uniform float fade_f; uniform vec2 anchor; uniform float rad; uniform float soft;
uniform vec2 vel; uniform float has_fg; uniform float rgbsplit; uniform vec2 size;
vec4 lay(sampler2D A, sampler2D T, vec2 p, float lod){
    vec2 uv = p / size;
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return vec4(0.0);
    vec4 a = textureLod(A, uv, lod); vec4 t = textureLod(T, uv, lod);
    return vec4(t.rgb * t.a + a.rgb * (1.0 - t.a), t.a + a.a * (1.0 - t.a));
}
vec4 hud(vec2 p){
    float off = texelFetch(rows, ivec2(int(clamp(p.y, 0.0, size.y - 1.0)), 0), 0).r;
    if (off > 90000.0) return vec4(0.0);
    p.x -= off;
    vec4 F = lay(fa, ft, p, 0.0) * fade_f;
    vec2 pb = (binv * vec3(p, 1.0)).xy;
    vec4 B = lay(ba, bt, pb, 0.75) * fade_b;
    B *= clamp((rad - length(pb - anchor)) / soft, 0.0, 1.0);
    if (has_fg > 0.5) B *= (1.0 - texture(fg, p / size).r);
    return F + B * (1.0 - F.a);
}
void main(){
    vec2 p = gl_FragCoord.xy;
    vec4 L = vec4(0.0);
    float sp = length(vel);
    if (sp > 4.0) {
        for (int k = 0; k < 5; k++) { float u = (float(k) / 4.0 - 0.5) * min(sp * 0.8, 53.0) / sp; L += hud(p + vel * u); }
        L /= 5.0;
    } else L = hud(p);
    if (rgbsplit > 0.5) { vec4 Lr = hud(p + vec2(rgbsplit, 0.0)); vec4 Lb = hud(p - vec2(rgbsplit, 0.0)); L.r = Lr.r; L.b = Lb.b; }
    vec3 b = texelFetch(base, ivec2(p), 0).rgb;
    float lum = luma(textureLod(base, p / size, 5.0).rgb);
    float Lf = clamp(0.80 + 0.55 * lum, 0.8, 1.3);
    vec3 o = b * (1.0 - L.a) + L.rgb * Lf;
    vec4 G = vec4(0.0);
    for (int k = 0; k < 4; k++) { vec2 d = vec2(cos(float(k) * 1.5708 + 0.4), sin(float(k) * 1.5708 + 0.4)) * 7.0; G += hud(p + d); }
    vec3 glow = clamp(G.rgb / 4.0 * 0.55 * Lf, 0.0, 1.0);
    o = 1.0 - (1.0 - clamp(o, 0.0, 1.0)) * (1.0 - glow);
    o_out(o);
}""".replace("o_out(o);", "o = vec4(o, 1.0);").replace("vec3 o = b", "vec3 oo = b").replace("o = 1.0 - (1.0 - clamp(o, 0.0, 1.0)) * (1.0 - glow);\n    o = vec4(o, 1.0);",
    "oo = 1.0 - (1.0 - clamp(oo, 0.0, 1.0)) * (1.0 - glow);\n    o = vec4(oo, 1.0);").replace("float Lf = clamp(0.80 + 0.55 * lum, 0.8, 1.3);\n    vec3 oo = b * (1.0 - L.a) + L.rgb * Lf;",
    "float Lf = clamp(0.80 + 0.55 * lum, 0.8, 1.3);\n    vec3 oo = b * (1.0 - L.a) + L.rgb * Lf;")


def _texm(slot, arr, mip=True):
    """带 mipmap 的纹理（局部亮度 / 景深模糊用）。"""
    import moderngl
    ctx = _S["ctx"]
    h, w = arr.shape[:2]
    comp = 1 if arr.ndim == 2 else arr.shape[2]
    key = ("m", slot, w, h, comp)
    t = _S["tex"].get(key)
    if t is None:
        t = ctx.texture((w, h), comp, dtype="f1")
        _S["tex"][key] = t
    t.write(np.ascontiguousarray(arr).tobytes())
    if mip:
        t.build_mipmaps()
        t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    else:
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t.repeat_x = False; t.repeat_y = False
    return t


def hud3(base, back_a, back_t, front_a, front_t, binv, anchor, rad, soft, fade_b, fade_f, fg=None, rows=None, vel=(0.0, 0.0), rgbsplit=0.0):
    import moderngl
    if "hud3" not in _S:
        ctx = _S["ctx"]
        vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        prog = ctx.program(vertex_shader=VS, fragment_shader=FS_HUD3)
        _S["hud3"] = (prog, ctx.simple_vertex_array(prog, vbo, "p"))
    ctx = _S["ctx"]
    if rows is None:
        rows = np.zeros(H, np.float32)
    key = ("rows", H)
    rt = _S["tex"].get(key)
    if rt is None:
        rt = ctx.texture((H, 1), 1, dtype="f4"); rt.filter = (moderngl.NEAREST, moderngl.NEAREST); _S["tex"][key] = rt
    rt.write(np.ascontiguousarray(rows, np.float32).tobytes())
    fgt = _texm(15, fg if fg is not None else np.zeros((4, 4), np.uint8), mip=False)
    texs = [("base", _texm(10, base)), ("ba", _texm(11, back_a)), ("bt", _texm(12, back_t)), ("fa", _texm(13, front_a, mip=False)),
            ("ft", _texm(14, front_t, mip=False)), ("fg", fgt), ("rows", rt)]
    m = np.asarray(binv, np.float32)
    return _run("hud3", texs, binv=tuple(m.T.flatten().tolist()), anchor=tuple(map(float, anchor)), rad=float(rad), soft=float(soft),
                fade_b=float(fade_b), fade_f=float(fade_f), vel=tuple(map(float, vel)), has_fg=1.0 if fg is not None else 0.0,
                rgbsplit=float(rgbsplit), size=(float(W), float(H)))

# -*- coding: utf-8 -*-
"""
HUD v5 组件库（用户 10-19："HUD 要做的更炫酷且富有机能感和设计感"；"HUD 的出现本身就是一套精密的 UI 设计以及融合精致的 motion 和前端设计艺术"）。

设计系统（token）
  网格   M = 96R 页边，U = 8R 基础间距；所有面板 / 文字按 U 对齐
  字阶   DISPLAY  Saira ExtraCondensed Black（巨型描边字 / 等级字母）
         H1 / H2  Rajdhani Bold / SemiBold（真名、数值）
         LABEL    Chakra Petch Medium，全大写，字距 0.22em（标签）
         MICRO    Share Tech Mono（极小的状态字）
         JP       思源黑 Heavy（日文名）；NP 用 Noto Serif Black（明朝 = Fate 宝具名的感觉）
  线重   hair 1.2R / line 2R / bold 4R；颜色角色：ink 白 0.92、dim 白 0.4、hair 白 0.25、accent = 画面主色、on-accent 近黑
动效（motion）
  每个元素有 (t0, dur, order)：入场按 t0 起、dur 内用 expo-out / back-out 缓动；退场按 order 反序（后进先出）在 ~10 帧内收回。
  线条：描边生长（stroke reveal）；面板：先画框线、再横向展开、再纵向展开、最后填充；文字：逐字解码（乱码 → 真字）或逐字上浮；
  量表：逐格步进点亮；环：分段依次出现 + 持续旋转；括号：飞入 + 回弹锁定 + 两次闪烁。
字体：Google Fonts（SIL OFL），assets/fonts/。
"""
import math, os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import style_frames as SF
from style_frames import W, H, R
from cards import cl
import hud_elems as HE
from hud_v2 import Canvas, SC, SH, _c, _t

ROOT = r"D:\A\provant"
FD = os.path.join(ROOT, "assets", "fonts")
M, U = 96 * R, 8 * R
INK, DIM, HAIR = 0.92, 0.40, 0.25
WHITE = (236, 242, 244)
ON_ACC = (14, 12, 14)
PANEL = (10, 12, 16)
FONT_FILES = {"display": "SairaExtraCondensed-Black.ttf", "display_b": "SairaExtraCondensed-Bold.ttf", "h1": "Rajdhani-Bold.ttf",
              "h2": "Rajdhani-SemiBold.ttf", "h3": "Rajdhani-Medium.ttf", "label": "ChakraPetch-Medium.ttf", "label_b": "ChakraPetch-Bold.ttf",
              "micro": "ShareTechMono-Regular.ttf"}
_F = {}
SCR = "0123456789ABCDEFHKMNPRSTUVXYZ#/<>+="


def fnt(kind, size):
    k = (kind, int(size))
    if k not in _F:
        if kind == "jp":
            _F[k] = SF.vfont(SF.F_JP, int(size), "Heavy")
        elif kind == "jp_m":
            _F[k] = SF.vfont(SF.F_JP, int(size), "Medium")
        elif kind == "np":
            _F[k] = SF.vfont(SF.F_SERIF_CJK, int(size), "Black")
        elif kind == "np_r":
            _F[k] = SF.vfont(SF.F_SERIF_CJK, int(size), "Regular")
        else:
            _F[k] = ImageFont.truetype(os.path.join(FD, FONT_FILES[kind]), int(size))
    return _F[k]


# ---------------------------------------------------------------- 缓动
def expo_out(x):
    x = cl(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def expo_io(x):
    x = cl(x)
    if x in (0.0, 1.0):
        return x
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2


def back_out(x, s=1.9):
    x = cl(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


class Clock:
    """元素时钟：入场 (t0, dur) + 退场（按 order 反序）。p() 返回 0..1 的显现度，ease 可选。"""
    def __init__(self, f, ex):
        self.f, self.ex = f, ex

    def p(self, t0, dur, order=0.5, ease=expo_out):
        k = ease((self.f - t0) / float(max(1, dur)))
        if self.ex > 0:
            k *= 1 - cl((self.ex - (1 - order) * 0.55) / 0.45)
        return k

    def raw(self, t0, dur):
        return cl((self.f - t0) / float(max(1, dur)))


# ---------------------------------------------------------------- 几何
def sector(cv, c, r0, r1, a0, a1, col, a=1.0, sy=1.0):
    """环形扇段（填充，抗锯齿），角度为度，sy = 纵向压缩（透视）。"""
    if a1 <= a0 or r1 <= r0 or a <= 0.01:
        return
    n = max(2, int((a1 - a0) / 1.5) + 1)
    angs = np.radians(np.linspace(a0, a1, n))
    o = np.stack([c[0] + np.cos(angs) * r1, c[1] + np.sin(angs) * r1 * sy], 1)
    i_ = np.stack([c[0] + np.cos(angs[::-1]) * r0, c[1] + np.sin(angs[::-1]) * r0 * sy], 1)
    HE.fpoly(cv, np.concatenate([o, i_]).tolist(), col, a)


def arc(cv, c, r, a0, a1, col, a=1.0, th=1.0, sy=1.0):
    if a1 <= a0 or a <= 0.01:
        return
    n = max(2, int((a1 - a0) / 1.0) + 1)
    angs = np.radians(np.linspace(a0, a1, n))
    pts = np.stack([c[0] + np.cos(angs) * r, c[1] + np.sin(angs) * r * sy], 1)
    HE.poly(cv, pts.tolist(), col, a, th, closed=False)


def grow_line(cv, p, q, k, col, a=1.0, th=1.0, from_mid=False):
    """描边生长：k = 0..1。from_mid = 从中点向两端长。"""
    if k <= 0.001 or a <= 0.01:
        return
    if from_mid:
        m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        p = (m[0] + (p[0] - m[0]) * k, m[1] + (p[1] - m[1]) * k)
        q = (m[0] + (q[0] - m[0]) * k, m[1] + (q[1] - m[1]) * k)
    else:
        q = (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k)
    cv.line(p, q, col, a, th)


def poly_grow(cv, pts, k, col, a=1.0, th=1.0, closed=True):
    """折线按总长生长。"""
    if k <= 0.001 or a <= 0.01:
        return
    P = list(pts) + ([pts[0]] if closed else [])
    segs = [(P[i], P[i + 1], math.hypot(P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1])) for i in range(len(P) - 1)]
    tot = sum(s[2] for s in segs); left = tot * k
    for p, q, L_ in segs:
        if left <= 0:
            break
        u = min(1.0, left / max(L_, 1e-6))
        cv.line(p, (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u), col, a, th)
        left -= L_


def notch_rect(x0, y0, x1, y1, ch, cut=(1, 0, 1, 0)):
    return HE.chamfer_rect(x0, y0, x1, y1, ch, cut)


def bracket(cv, box, L_, th, col, a=1.0, notch=None):
    """四角括号；notch = 角上 45° 小切口。"""
    x0, y0, x1, y1 = box
    n = notch if notch is not None else th * 2.2
    for sx, sy, x, y in ((1, 1, x0, y0), (-1, 1, x1, y0), (-1, -1, x1, y1), (1, -1, x0, y1)):
        HE.poly(cv, [(x + sx * L_, y), (x + sx * n, y), (x, y + sy * n), (x, y + sy * L_)], col, a, th, closed=False)


def ticks(cv, c, r, n, L_major, L_minor, rot, col, a, th, every=5, a0=0, a1=360, sy=1.0):
    for j in range(n):
        ang = a0 + (a1 - a0) * j / n + rot
        u = math.radians(ang)
        L_ = L_major if j % every == 0 else L_minor
        p = (c[0] + math.cos(u) * r, c[1] + math.sin(u) * r * sy)
        q = (c[0] + math.cos(u) * (r + L_), c[1] + math.sin(u) * (r + L_) * sy)
        cv.line(p, q, col, a, th)


def seg_meter(cv, x, y, w, h, n, lit, col, a=1.0, gap=None):
    """分格量表：n 格，lit 格（可为小数，最后一格按比例）。"""
    gap = gap if gap is not None else max(2.0, 0.35 * w / n)
    bw = (w - gap * (n - 1)) / n
    for j in range(n):
        bx = x + j * (bw + gap)
        if j < int(lit):
            cv.rect((bx, y), (bx + bw, y + h), col, a)
        elif j < lit:
            cv.rect((bx, y), (bx + bw * (lit - j), y + h), col, a)
            cv.rect((bx, y), (bx + bw, y + h), WHITE, HAIR * a, 1.0 * R)
        else:
            cv.rect((bx, y), (bx + bw, y + h), WHITE, HAIR * a, 1.0 * R)


def stripes(cv, x0, y0, x1, y1, col, a, pitch=None, ang=45):
    """斜纹填充（面板标题条 / 警示块）。"""
    pitch = pitch or 10 * R
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    lay = np.zeros((int(y1 - y0) + 2, int(x1 - x0) + 2), np.uint8)
    hh, ww = lay.shape
    for k in range(-hh, ww + hh, int(pitch)):
        cv2.line(lay, (k, hh), (k + hh, 0), 255, max(1, int(pitch * 0.32)), cv2.LINE_AA)
    m = lay.astype(np.float32)[..., None] / 255 * a
    reg = cv.a[int(y0):int(y0) + hh, int(x0):int(x0) + ww].astype(np.float32)
    hh2, ww2 = reg.shape[:2]
    m = m[:hh2, :ww2]
    col4 = np.array(_c(col, 1.0), np.float32)
    cv.a[int(y0):int(y0) + hh2, int(x0):int(x0) + ww2] = np.clip(reg * (1 - m) + col4 * m, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 文字
def text_w(s, f, track=0.0):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    return sum(d.textlength(ch, font=f) for ch in s) + track * max(0, len(s) - 1)


def ttext(cv, xy, s, f, col, a=1.0, track=0.0, align="l", decode=1.0, seed=0, rise=None):
    """带字距的文字。decode < 1：已解码的字之后显示乱码（拉丁字母），CJK 逐字出现；rise = 每字上浮入场 (k 0..1)。"""
    if a <= 0.01 or not s:
        return 0.0
    w_ = text_w(s, f, track)
    x, y = xy
    if align == "r":
        x -= w_
    elif align == "c":
        x -= w_ / 2
    n = len(s)
    shown = decode * (n + 2)
    rng = np.random.default_rng(seed)
    d = cv.d
    for i, ch in enumerate(s):
        cw = d.textlength(ch, font=f)
        if i < shown - 2:
            c_ = ch; al = a
        elif i < shown and ch != " ":
            if ord(ch) > 0x2E80:
                c_ = ch; al = a * 0.45
            else:
                c_ = SCR[int(rng.integers(len(SCR)))]; al = a * 0.75
        else:
            c_ = None; al = 0
        if c_ is not None:
            yy = y
            if rise is not None:
                kk = cl(rise * (n + 4) - i)
                yy = y + (1 - expo_out(kk)) * f.size * 0.5
                al *= kk
            d.text((x, yy), c_, font=f, fill=tuple(col) + (int(255 * al),))
        x += cw + track
    return w_


def outline_text(s, f, col, a, stroke, fill=None, fill_k=0.0, track=0.0):
    """巨型描边字（返回 RGBA 图）：fill_k = 从左到右实心填充的比例（强调色扫过）。"""
    w_ = int(text_w(s, f, track) + stroke * 4 + 8)
    asc, desc = f.getmetrics()
    h_ = asc + desc + int(stroke * 4) + 8
    im = Image.new("RGBA", (w_, h_), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    x = stroke * 2
    for ch in s:
        d.text((x, stroke * 2), ch, font=f, fill=(0, 0, 0, 0), stroke_width=int(stroke), stroke_fill=tuple(col) + (int(255 * a),))
        x += d.textlength(ch, font=f) + track
    if fill is not None and fill_k > 0:
        m = Image.new("L", (w_, h_), 0); dm = ImageDraw.Draw(m); x = stroke * 2
        for ch in s:
            dm.text((x, stroke * 2), ch, font=f, fill=255)
            x += dm.textlength(ch, font=f) + track
        arr = np.array(m); arr[:, int(w_ * fill_k):] = 0
        fl = Image.new("RGBA", (w_, h_), tuple(fill[:3]) + (255,))
        fl.putalpha(Image.fromarray((arr.astype(np.float32) * fill[3] / 255.0).astype(np.uint8)))
        im.alpha_composite(fl)
    return im


def vtext(cv, x, y, s, f, col, a, k=1.0, step=None):
    """竖排（日文）：逐字下落出现。"""
    step = step or f.size * 1.08
    n = len(s)
    for i, ch in enumerate(s):
        kk = cl(k * (n + 3) - i)
        if kk <= 0:
            break
        cv.d.text((x, y + i * step - (1 - expo_out(kk)) * 10 * R), ch, font=f, fill=tuple(col) + (int(255 * a * kk),))
    return n * step

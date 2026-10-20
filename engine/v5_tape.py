# -*- coding: utf-8 -*-
"""警戒线（封锁胶带）转场层（用户 10-20："人物转场除了人物边框外还可以加一些警戒线等画面处理方式（由你来具体设计表现形式）"）。
设计：
  · 胶带 = 角色主色底 + 45° 黑色危险斜纹段，与文字段交替；文字只写核对过的信息（CAUTION + 真名 / 职阶 / 日文名，来自 EDL 的 tape 标签）。
  · 运动：切点前 2 条胶带沿各自斜角从画外甩入（expo_out，错开 2 帧）→ 切点整体一顿（2 帧上下抖 + 胶带层 RGB 错位）→
    （cross 模式）切后第三条反色大字条拍入 → 收尾时沿自身方向被扯走（expo_in，方向交替），两端是撕开的锯齿口。
  · 融入画面（不要"像两个图层"）：胶带投影（模糊 6R、偏移 8R/10R、45% 黑）、移动时沿轴向运动模糊、胶带在时画面整体压暗 12%、
    文字段一直慢速滚动；颜色跟着入场角色的配色走（acc，已避开橙色）。
  · slit 模式：两条胶带横跨上下黑边（像封锁的门缝），中间缝里的名字不挡；随缝张开被扯走。
"""
import math, os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import style_frames as SF
from style_frames import W, H, R, font
from cards import cl

F_DISP = os.path.join(r"D:\A\provant", "assets", "fonts", "SairaExtraCondensed-Black.ttf")
INK, PAPER = (10, 10, 12), (246, 244, 238)


def eo(x):
    x = cl(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ei(x):
    x = cl(x)
    return 0.0 if x <= 0 else 2 ** (10 * (x - 1))


def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _cjk(s):
    return any(ord(ch) >= 0x3000 for ch in s)


_STRIP = {}


def strip(labels, col, h, length, inverse=False):
    """横向胶带 RGBA（length × h）：[斜纹][⚠ 文字][斜纹][文字]… 循环铺满；两端锯齿撕口。"""
    key = (tuple(labels), tuple(col), h, length, inverse)
    if key in _STRIP:
        return _STRIP[key]
    bg = (16, 16, 18) if inverse else tuple(int(v) for v in col)
    st = tuple(int(v) for v in col) if inverse else (12, 12, 14)
    ink = tuple(int(v) for v in col) if inverse else (INK if _lum(col) > 110 else PAPER)
    im = Image.new("RGBA", (length, h), bg + (255,))
    d = ImageDraw.Draw(im)
    fd = font(F_DISP, int(h * 0.66))
    fj = SF.vfont(SF.F_JP, int(h * 0.50), "Heavy")
    p = h * 0.72                                            # 斜纹周期
    x, i = 0.0, 0
    while x < length:
        sw = h * (1.5 if i % 2 == 0 else 0.9)
        for k in range(-3, int(sw / p) + 3):                # 斜纹段（裁在段内）
            x0 = x + k * p
            poly = [(x0, h), (x0 + p / 2, h), (x0 + p / 2 + h, 0), (x0 + h, 0)]
            poly = [(min(max(px, x), x + sw), py) for px, py in poly]
            d.polygon(poly, fill=st + (255,))
        x += sw + h * 0.35
        lab = labels[i % len(labels)]
        if lab == "CAUTION":                                # 警示三角
            t = h * 0.62; cy = h / 2
            d.polygon([(x, cy + t * 0.5), (x + t * 0.58, cy - t * 0.52), (x + t * 1.16, cy + t * 0.5)], outline=ink + (255,), width=max(2, int(h * 0.07)))
            d.text((x + t * 0.58, cy + t * 0.08), "!", font=font(F_DISP, int(t * 0.8)), fill=ink + (255,), anchor="mm")
            x += t * 1.16 + h * 0.3
        f_ = fj if _cjk(lab) else fd
        d.text((x, h / 2 + (0 if _cjk(lab) else h * 0.03)), lab, font=f_, fill=ink + (255,), anchor="lm")
        x += d.textlength(lab, font=f_) + h * 0.35
        i += 1
    lt = max(2, int(h * 0.06))                              # 上下描边 + 一条高光
    d.rectangle([0, 0, length, lt], fill=INK + (255,)); d.rectangle([0, h - lt, length, h], fill=INK + (255,))
    d.line([(0, lt + 2), (length, lt + 2)], fill=(255, 255, 255, 70), width=max(1, int(R)))
    a = np.asarray(im).copy()
    rng = np.random.default_rng(len(labels) * 7 + h)
    tooth = max(6, int(h * 0.22))
    for end in (0, 1):                                      # 撕口：两端随机锯齿
        ys = np.arange(h)
        prof = np.interp(ys, np.linspace(0, h, 9), rng.uniform(0.15, 1.0, 9)) * tooth
        for y in range(h):
            c = int(prof[y])
            if end == 0:
                a[y, :c, 3] = 0
            else:
                a[y, length - c:, 3] = 0
    _STRIP[key] = a
    return a


def _specs(mode, pre, n, seed):
    rng = np.random.default_rng(seed)
    j = lambda s: float(rng.uniform(-s, s))
    post = n - pre
    if mode == "slit":
        t_out = pre + int(0.40 * post)
        return [dict(y=0.20 + j(0.03), ang=-6.5 + j(2.0), h=0.074, d=1, t_in=max(0, pre - 10), t_out=t_out),
                dict(y=0.81 + j(0.03), ang=5.0 + j(2.0), h=0.068, d=-1, t_in=max(0, pre - 8), t_out=t_out + 2)]
    t_out = pre + int(0.55 * post)
    s = 1 if rng.random() < 0.5 else -1
    return [dict(y=0.30 + j(0.04), ang=s * (-11 + j(2.5)), h=0.082, d=1, t_in=0, t_out=t_out),
            dict(y=0.70 + j(0.04), ang=s * (8 + j(2.5)), h=0.074, d=-1, t_in=2, t_out=t_out + 2),
            dict(y=0.50 + j(0.03), ang=s * (-3 + j(1.5)), h=0.118, d=1, t_in=pre + 1, t_out=t_out + 4, big=True)]


def tapes(img, f, n, pre, acc, labels, seed=0, mode="cross"):
    """在转场帧 img 上叠胶带（f = 转场内帧号，pre = 切点帧号）。labels：第一条 = ["CAUTION", 真名, 职阶, 日文名…]。"""
    specs = _specs(mode, pre, n, seed)
    L = int(math.hypot(W, H) * 1.25)
    lay = np.zeros((H, W, 4), np.float32)
    vis = 0.0
    for k, sp in enumerate(specs):
        if f < sp["t_in"]:
            continue
        h = int(sp["h"] * H)

        def off(ff):
            o = (1 - eo((ff - sp["t_in"]) / 7.0)) * 1.35 * W * sp["d"]
            if ff >= sp["t_out"]:
                o -= ei((ff - sp["t_out"]) / 8.0) * 1.7 * W * sp["d"] * (1 if k % 2 == 0 else -1)
            return o
        o, v = off(f), off(f) - off(f - 1)
        if abs(o) > 1.6 * W:
            continue
        labs = [x for x in labels if x] if not sp.get("big") else [x for x in labels[1:] if x][:2] or labels
        S = strip(labs, acc["main"], h, L + int(420 * R), inverse=bool(sp.get("big")))
        sc = int((f * 7 * R * (1 if k % 2 == 0 else -1)) % int(400 * R))
        win = S[:, sc:sc + L].astype(np.float32)
        kb = int(min(abs(v) * 0.55, 140 * R))
        if kb >= 3:                                          # 沿轴向运动模糊
            win = cv2.blur(win, (kb, 1))
        a = math.radians(sp["ang"]); ca, sa = math.cos(a), math.sin(a)
        cx = W / 2 + o * ca; cy = sp["y"] * H + o * sa
        if pre <= f < pre + 2:                              # 切点一顿
            cy += (6 * R, -4 * R)[f - pre]
        M = np.float32([[ca, -sa, cx - (ca * L / 2 - sa * h / 2)], [sa, ca, cy - (sa * L / 2 + ca * h / 2)]])
        t = cv2.warpAffine(win, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        al = t[..., 3:4] / 255.0
        lay[..., :3] = lay[..., :3] * (1 - al) + t[..., :3] * al
        lay[..., 3:4] = lay[..., 3:4] + al * (1 - lay[..., 3:4])
        vis = max(vis, 1.0 - min(1.0, abs(o) / W))
    if vis <= 0.0:
        return img
    if pre <= f < pre + 2:                                  # 切点：胶带层 RGB 错位
        dx = int(5 * R)
        lay[..., 0] = np.roll(lay[..., 0], dx, axis=1); lay[..., 2] = np.roll(lay[..., 2], -dx, axis=1)
    out = img.astype(np.float32) * (1 - 0.12 * vis)        # 胶带在时画面压暗
    sh = cv2.GaussianBlur(lay[..., 3], (0, 0), 6 * R)
    sh = np.roll(np.roll(sh, int(10 * R), axis=0), int(8 * R), axis=1)
    out *= (1 - 0.45 * sh[..., None])
    al = lay[..., 3:4]
    out = out * (1 - al) + lay[..., :3]                     # lay 是预乘色
    return np.clip(out, 0, 255).astype(np.uint8)

# -*- coding: utf-8 -*-
"""
美漫风格（用户 10-19："可以用一些美漫风格化处理"）。
v4（看过 v5 成片后："有些漫画风格处理还是会导致人物和画面失真（像奶油一样化开）"；
    "美漫风格的字符可以再个性化一些（参考蜘蛛侠纵横宇宙的美学）"）：
  画面   全分辨率；色板平涂只铺在"平坦区"（局部纹理低的地方），脸 / 发丝 / 衣褶这些有纹理的地方保留原图 → 不再化开；
         原图提彩 + 高频细节回加 + 原画细描线 + 大轮廓 XDoG；暗面 Ben-Day 圆点；整帧 CMYK 套印错位（青 / 品版反向偏 2–6 px）
  字     蜘蛛侠：纵横宇宙式手写美漫字：
         说明框  手裁不规则黄框（四角各自抖动、倾斜 ±4°）+ 品红套印错版影 + 青色描边错版 + 框内细网点；
                 名字 Comic Neue Bold 逐字蹦出（每字随机 ±3° 旋转 / 基线抖动），右下角黑底 Bungee 职阶小标
         拟声字  Bungee 方块字逐字弹出（回弹 + 各自倾斜）：斜向立体挤出（角色暗色）→ 粗黑描边 → 角色色渐变填充 + 下半 Ben-Day 网点
                 → 青 / 品红错版影；背后锯齿爆炸气球（白底 + 角色色网点 + 黑边）
字体：Google Fonts（Bungee / Comic Neue 为 SIL OFL）。
"""
import math, os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from style_frames import W, H, R
from cards import eo, bo, cl

ROOT = r"D:\A\provant"
FD = os.path.join(ROOT, "assets", "fonts")
F_BANG = os.path.join(FD, "Bangers-Regular.ttf")
F_BUNGEE = os.path.join(FD, "Bungee-Regular.ttf")
F_LETTER = os.path.join(FD, "ComicNeue-Bold.ttf")
INK = np.array((16, 14, 20), np.float32)
PAPER = np.array((247, 240, 222), np.float32)
YEL = (255, 230, 60)
MAG = (236, 40, 124)
CYA = (30, 190, 232)
_PAL, _DOT, _PAPER, _FT = {}, {}, {}, {}


def ft(path, size):
    k = (path, int(size))
    if k not in _FT:
        _FT[k] = ImageFont.truetype(path, int(size))
    return _FT[k]


def _paper(h, w):
    if (h, w) not in _PAPER:
        rng = np.random.default_rng(7)
        n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2)
        fib = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=9, sigmaY=0.8)
        _PAPER[(h, w)] = (n * 3.0 + fib * 5.0)[..., None]
    return _PAPER[(h, w)]


def _screen(h, w, pitch, ang, dx=0.0, dy=0.0):
    k = (h, w, round(pitch, 2), ang, round(dx, 1), round(dy, 1))
    if k not in _DOT:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx += dx; yy += dy
        t = math.radians(ang)
        u = (xx * math.cos(t) + yy * math.sin(t)) / pitch
        v = (-xx * math.sin(t) + yy * math.cos(t)) / pitch
        _DOT[k] = np.hypot(u - np.round(u), v - np.round(v))
    return _DOT[k]


def palette(key, lab_small, k=8):
    """每镜固定色板（Lab k-means），按亮度排序。画面中部多采样一倍。"""
    if key not in _PAL:
        h, w = lab_small.shape[:2]
        mid = lab_small[int(h * 0.15):int(h * 0.85), int(w * 0.25):int(w * 0.75)]
        X = np.concatenate([lab_small.reshape(-1, 3), mid.reshape(-1, 3), mid.reshape(-1, 3)]).astype(np.float32)
        X = X[np.random.default_rng(1).choice(len(X), min(len(X), 24000), replace=False)]
        _, lab, cen = cv2.kmeans(X, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3, cv2.KMEANS_PP_CENTERS)
        _PAL[key] = cen[np.argsort(cen[:, 0])]
    return _PAL[key]


def xdog(gray, sigma=1.0, k=1.6, p=22.0, eps=-0.012, phi=60.0):
    g = gray.astype(np.float32) / 255.0
    a = cv2.GaussianBlur(g, (0, 0), sigma); b = cv2.GaussianBlur(g, (0, 0), sigma * k)
    s = (1 + p) * a - p * b
    e = np.where(s >= eps, 1.0, 1.0 + np.tanh(phi * (s - eps)))
    return 1.0 - np.clip(e, 0, 1)


def _clean(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


# ---------------------------------------------------------------- 字（纵横宇宙式）
def _dots_img(w, h, pitch, col, a, r_frac=0.34):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx + yy) / pitch; v = (yy - xx) / pitch
    d = np.hypot(u - np.round(u), v - np.round(v))
    m = (d < r_frac).astype(np.float32) * a
    im = np.zeros((h, w, 4), np.uint8); im[..., :3] = col; im[..., 3] = (m * 255).astype(np.uint8)
    return Image.fromarray(im, "RGBA")


def _poly_rot(pts, c, ang):
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    return [(c[0] + (x - c[0]) * ca - (y - c[1]) * sa, c[1] + (x - c[0]) * sa + (y - c[1]) * ca) for x, y in pts]


def _masked(im, pts, ox, oy):
    m = Image.new("L", im.size, 0); ImageDraw.Draw(m).polygon([(x - ox, y - oy) for x, y in pts], fill=255)
    im.putalpha(Image.fromarray((np.asarray(im.getchannel("A")).astype(np.float32) * np.asarray(m) / 255).astype(np.uint8)))
    return im


def caption_sv(o, name, tag, f, pal, seed, pos):
    """手裁说明框：错版影 + 不规则黄框 + 框内网点 + 逐字蹦出的手写字 + 黑底职阶小标。o = RGBA 画面（就地画）。"""
    rng = np.random.default_rng(seed)
    fl = ft(F_LETTER, 50 * R)
    d0 = ImageDraw.Draw(o)
    ws = [d0.textlength(ch, font=fl) for ch in name]
    tw = sum(ws) + 2 * R * len(name)
    bw, bh = tw + 64 * R, 92 * R
    x0, y0 = pos
    ang = float(rng.uniform(-4, 4))
    jits = rng.uniform(-7, 7, 8) * R
    rots = rng.uniform(-3, 3, len(name)); bases = rng.uniform(-3, 3, len(name)) * R
    k = eo((f - 2) / 6.0)
    if k <= 0.01:
        return
    box = [(x0 + jits[0], y0 + jits[1]), (x0 + bw * k + jits[2], y0 + jits[3]), (x0 + bw * k + jits[4], y0 + bh + jits[5]), (x0 + jits[6], y0 + bh + jits[7])]
    c = (x0 + bw / 2, y0 + bh / 2)
    box = _poly_rot(box, c, ang)
    lay = Image.new("RGBA", o.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.polygon([(x + 11 * R, y + 9 * R) for x, y in box], fill=MAG + (220,))                     # 品红错版影
    d.polygon(box, fill=YEL + (255,))
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    dots = _masked(_dots_img(int(max(xs) - min(xs)) + 2, int(max(ys) - min(ys)) + 2, 7 * R, (160, 110, 0), 0.22), box, min(xs), min(ys))
    lay.alpha_composite(dots, (int(min(xs)), int(min(ys))))
    d.line(box + [box[0]], fill=(16, 14, 20, 255), width=max(3, int(4.5 * R)), joint="curve")
    d.line([(x - 5 * R, y - 4 * R) for x, y in box + [box[0]]], fill=CYA + (190,), width=max(1, int(2 * R)))  # 青色描边错版
    if k > 0.9:                                             # 逐字蹦出（每 1.5 帧一个），每字随机倾斜 / 基线
        x = x0 + 32 * R
        for i, (ch, w_) in enumerate(zip(name, ws)):
            kk = cl((f - 6 - i * 1.5) / 4.0)
            if kk <= 0:
                break
            if ch != " ":
                s = 0.6 + 0.4 * bo(kk, 2.2)
                gi = Image.new("RGBA", (int(w_ + 20 * R), int(80 * R)), (0, 0, 0, 0))
                ImageDraw.Draw(gi).text((10 * R, 6 * R), ch, font=fl, fill=(16, 14, 20, 255))
                gi = gi.rotate(float(rots[i]) - ang, resample=Image.BICUBIC, expand=False)
                gi = gi.resize((max(1, int(gi.width * s)), max(1, int(gi.height * s))), Image.BICUBIC)
                p = _poly_rot([(x + w_ / 2, y0 + bh / 2 + bases[i])], c, ang)[0]
                lay.alpha_composite(gi, (int(p[0] - gi.width / 2), int(p[1] - gi.height / 2)))
            x += w_ + 2 * R
    if tag and f >= 10:                                     # 黑底 Bungee 职阶小标（压在框右下角，反向倾斜）
        fb = ft(F_BUNGEE, 26 * R)
        tw2 = d.textlength(tag, font=fb)
        ti = Image.new("RGBA", (int(tw2 + 36 * R), int(46 * R)), (16, 14, 20, 255))
        ImageDraw.Draw(ti).text((18 * R, 6 * R), tag, font=fb, fill=tuple(pal["main"]) + (255,))
        s = bo(cl((f - 10) / 5.0), 2.4)
        ti = ti.resize((max(1, int(ti.width * s)), max(1, int(ti.height * s))), Image.BICUBIC).rotate(-ang * 1.6 + 3, resample=Image.BICUBIC, expand=True)
        lay.alpha_composite(ti, (int(x0 + bw - ti.width * 0.6), int(y0 + bh - 14 * R)))
    o.alpha_composite(lay)


_LET = {}


def _sfx_letter(ch, fs, pal, depth):
    """单个拟声字母：立体挤出 → 黑描边 → 渐变填充 + 下半网点 → 青 / 品错版影（按字母 + 颜色缓存）。"""
    key = (ch, fs.size, tuple(pal["main"]), tuple(pal["hi"]), tuple(pal["dk"]), depth)
    if key in _LET:
        return _LET[key]
    tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
    w_ = int(tmp.textlength(ch, font=fs)); asc, desc = fs.getmetrics()
    pad = int(40 * R)
    Wd, Hd = w_ + 2 * pad + depth, asc + desc + 2 * pad + depth
    m = Image.new("L", (Wd, Hd), 0); ImageDraw.Draw(m).text((pad, pad), ch, font=fs, fill=255)
    ol = Image.new("L", (Wd, Hd), 0); ImageDraw.Draw(ol).text((pad, pad), ch, font=fs, fill=255, stroke_width=int(9 * R), stroke_fill=255)
    out = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
    for col, dx, dy in ((CYA, -9, -7), (MAG, 9, 7)):
        sh = Image.new("RGBA", (Wd, Hd), col + (255,)); sh.putalpha(ol.point(lambda v: int(v * 0.8)))
        out.alpha_composite(sh, (int(dx * R), int(dy * R)))
    dk = tuple(int(v * 0.55) for v in pal["dk"])
    for s_ in range(depth, 0, -2):
        ex = Image.new("RGBA", (Wd, Hd), dk + (255,)); ex.putalpha(ol)
        out.alpha_composite(ex, (s_, s_))
    ink = Image.new("RGBA", (Wd, Hd), (16, 14, 20, 255)); ink.putalpha(ol)
    out.alpha_composite(ink)
    t_ = np.linspace(0, 1, Hd)[:, None, None]
    c0, c1 = np.array(pal["hi"], np.float32), np.array(pal["main"], np.float32)
    grad = np.broadcast_to(c0 * (1 - t_) + c1 * t_, (Hd, Wd, 3)).copy()
    dd = _screen(Hd, Wd, 7 * R, 45.0) < 0.30
    lower = np.broadcast_to(np.linspace(0, 1, Hd)[:, None] > 0.55, (Hd, Wd))
    grad[dd & lower] = grad[dd & lower] * 0.55
    fl = np.dstack([grad, np.asarray(m)[..., None]]).astype(np.uint8)
    out.alpha_composite(Image.fromarray(fl, "RGBA"))
    _LET[key] = out
    return out


def sfx_sv(o, text, f, pal, seed, center):
    """拟声字：锯齿网点气球 + 逐字弹出的 Bungee 立体字。"""
    rng = np.random.default_rng(seed)
    fs = ft(F_BUNGEE, 190 * R)
    widths = [ImageDraw.Draw(Image.new("L", (1, 1))).textlength(ch, font=fs) for ch in text]
    tw = sum(widths) * 0.78 + 6 * R * (len(text) - 1)
    pj = rng.uniform(-0.06, 0.06, 36); pr = rng.uniform(-0.10, 0.10, 36)
    rots = rng.uniform(-9, 9, len(text)); dys = rng.uniform(-14, 14, len(text))
    kb = bo(cl((f - 1) / 6.0), 2.0)
    if kb > 0.02:
        n = 18; bw_, bh_ = tw * 0.62 + 120 * R, 230 * R
        pts = []
        for j in range(n * 2):
            a = math.pi * 2 * j / (n * 2) + pj[j]
            rr = (1.0 if j % 2 == 0 else 0.70) * (1 + pr[j]) * kb
            pts.append((center[0] + math.cos(a) * bw_ * rr, center[1] + math.sin(a) * bh_ * rr))
        lay = Image.new("RGBA", o.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        d.polygon([(x + 12 * R, y + 12 * R) for x, y in pts], fill=(16, 14, 20, 210))
        d.polygon(pts, fill=(252, 248, 236, 255))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        dots = _masked(_dots_img(int(max(xs) - min(xs)) + 2, int(max(ys) - min(ys)) + 2, 10 * R, pal["main"], 0.85, 0.30), pts, min(xs), min(ys))
        lay.alpha_composite(dots, (int(min(xs)), int(min(ys))))
        d.line(pts + [pts[0]], fill=(16, 14, 20, 255), width=max(3, int(6 * R)), joint="curve")
        o.alpha_composite(lay)
    x = center[0] - tw / 2
    for i, (ch, w_) in enumerate(zip(text, widths)):
        kk = cl((f - 2 - i * 1.5) / 5.0)
        if kk > 0.01 and ch != " ":
            im = _sfx_letter(ch, fs, pal, int(16 * R))
            s = 0.2 + 0.8 * bo(kk, 2.6)
            g = im.rotate(float(rots[i]), resample=Image.BICUBIC, expand=True)
            g = g.resize((max(1, int(g.width * s * 0.78)), max(1, int(g.height * s * 0.78))), Image.BICUBIC)
            cy = center[1] + (dys[i] - (i % 2) * 10) * R
            o.alpha_composite(g, (int(x + w_ * 0.39 - g.width / 2), int(cy - g.height / 2)))
        x += w_ * 0.78 + 6 * R


# ---------------------------------------------------------------- 画面
def comic_us(rgb, f, acc, seed, caption=None, sfx=None, key=None, tag=None):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pal = palette(key if key is not None else seed, cv2.resize(lab, (320, 180), interpolation=cv2.INTER_AREA))
    small = cv2.resize(lab, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    lbl = cv2.medianBlur(((small[:, :, None, :] - pal[None, None]) ** 2).sum(-1).argmin(-1).astype(np.uint8), 3)
    col = pal[cv2.resize(lbl, (W, H), interpolation=cv2.INTER_NEAREST)].copy()
    col[..., 1:] = 128 + (col[..., 1:] - 128) * 1.45
    col[..., 0] = np.clip(col[..., 0] * 0.55 + 105, 0, 255)
    flat = cv2.GaussianBlur(cv2.cvtColor(np.clip(col, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32), (0, 0), 1.2)
    lab2 = lab.copy(); lab2[..., 1:] = np.clip(128 + (lab2[..., 1:] - 128) * 1.15, 0, 255)          # v5：提彩减半（"还是有很多画面失真"）
    vivid = cv2.cvtColor(lab2.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (9, 9)); tex = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - mu * mu, 0))
    w_flat = (0.22 * (1 - np.clip(tex / 6.0, 0, 1)))[..., None]        # 只在很平的区域轻压色：其余保持原图（不化开、不失真）
    base = vivid * (1 - w_flat) + flat * w_flat
    Ls = cv2.GaussianBlur(lab[..., 0], (0, 0), 2.0) / 255.0
    thr = float(np.clip(np.percentile(Ls[::4, ::4], 34), 0.20, 0.52))
    shadow = cv2.GaussianBlur(cv2.morphologyEx((Ls < thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 1.0)
    dots = (_screen(H, W, 5.0 * R, 45.0) < 0.36).astype(np.float32)
    base = base * (1 - shadow[..., None] * dots[..., None] * 0.24)
    base = base + (g - cv2.GaussianBlur(g, (0, 0), 1.6))[..., None] * 0.45
    import style_frames as SF_
    L_ = SF_.lineart(rgb, thick=1)
    edge = _clean(xdog(g.astype(np.uint8), sigma=1.6, p=26.0) > 0.6, int(220 * R)).astype(np.float32)
    ln = np.clip(L_ * 0.80 + cv2.GaussianBlur(edge, (0, 0), 0.7) * 0.60, 0, 1)
    out = base * (1 - ln[..., None]) + INK * ln[..., None]
    out = np.clip(out * 0.97 + PAPER * 0.03 + _paper(H, W) * 0.4, 0, 255).astype(np.uint8)
    sh = int(round(6.0 * R * (1 - cl(f / 8.0))))          # CMYK 套印错位只在入场 8 帧（常驻的 2 px 错位会被看成失真）
    if sh:
        out[..., 0] = np.roll(out[..., 0], sh, axis=1); out[..., 2] = np.roll(out[..., 2], -sh, axis=1)
    m = int(10 * R)
    out[:m] = 246; out[-m:] = 246; out[:, :m] = 246; out[:, -m:] = 246
    cv2.rectangle(out, (m, m), (W - m - 1, H - m - 1), (16, 14, 20), max(3, int(4 * R)), cv2.LINE_AA)
    if caption or sfx:
        o = Image.fromarray(out).convert("RGBA")
        if caption:
            caption_sv(o, caption, tag, f, acc, seed, (m + 44 * R, m + 40 * R))
        if sfx and f >= 1:
            rng = np.random.default_rng(seed)
            sfx_sv(o, sfx, f, acc, seed, (W * (0.27 if rng.random() < 0.5 else 0.73), H * 0.70))
        out = np.asarray(o.convert("RGB"))
    return np.ascontiguousarray(out)

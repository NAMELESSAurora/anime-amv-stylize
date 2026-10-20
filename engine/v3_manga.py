# -*- coding: utf-8 -*-
"""
漫画分镜风格（用户 10-19 第 2 条："引入一些先前的漫画分镜风格化处理方式（grokbot 的和你自己学习的）"）。
来源：research/manga_bw_g11_style（G11：格内去色灰阶 + 格外纸面 AM 网点、四边渐浓中心干净、网点固定在纸面上）
     + v1 风格帧 D_IMPACT（三格：斜边主格 + 眼部特写格 + 红墨小格，竖排说明框，集中线）
     + v1 风格帧 B_POSTER（riso 双色海报：角色亮相卡）。
版式（每镜按种子 / 时长挑一个）：
  band    G11：一条横贯的宽格 + 竖排说明框（日文歌词 / 角色名）
  impact  D：斜边主格（集中线，hold 2 重画）+ 右上特写格 + 右下强调色墨格
  story   分镜：上宽格 + 下三小格；每格在它出现的那一刻定格（右→左读，像一页漫画被逐格画出来），最后一格是活的
  slant   两格对角
格内：gray（G11 灰阶，对比 1.1，轻颗粒；抒情）/ ink（描线 + 阴影网点，动作）；强调色只给一个墨格。
运动：格逐个砸入（1.12→1 back-out + 从格的方向滑入 + 1.5° 回正），格内画面缓推 1.0→1.06，集中线 hold 2 重画；纸面网点不动。
riso：角色亮相海报（红版 / 墨版错位滑入套准、主字从基线升起），画面 12 fps 步进。
"""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw

import style_frames as SF
from style_frames import W, H, R, font, gray, lineart
from cards import eo, bo, cl, halftone_c

PAPER = 246
INK = 14
_PT = {}


def paper_tone(key, holes, smooth=False):
    """格外纸面：AM 网点（45°，20 px@4K → 13 px@1440），径向渐变 0.45→1.15 smoothstep，格内挖空。缓存。
    smooth=True：同一渐变改连续调（纸色 → 墨色），不出网点。"""
    if (key, smooth) in _PT:
        return _PT[(key, smooth)]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rn = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / math.sqrt(2) * 1.25
    t = np.clip((rn - 0.45) / (1.15 - 0.45), 0, 1); t = t * t * (3 - 2 * t)
    if smooth:
        page = (PAPER * (1 - 0.85 * t) + INK * 0.85 * t).astype(np.uint8)
        _PT[(key, smooth)] = page
        return page
    dens = (t * 0.85 * 255).astype(np.uint8)
    dots = halftone_c(255 - dens, 13 * R * 0.75, 45)             # 0 = 墨
    page = np.where(dots == 0, INK, PAPER).astype(np.uint8)
    _PT[(key, smooth)] = page
    return page


def grain(shape, seed, std=6.0):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, std, (shape[0] // 2 + 1, shape[1] // 2 + 1)).astype(np.float32)
    return cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def tone_gray(img, seed):
    g = gray(img).astype(np.float32)
    lo, hi = np.percentile(g[::4, ::4], [2, 98])                 # 先拉开色阶：暗镜头不发闷
    g = np.clip((g - lo) / max(1.0, hi - lo), 0, 1) * 215 + 20
    g = np.clip((g - 128) * 1.1 + 128 + grain(g.shape, seed, 5.0), 0, 255)
    return g.astype(np.uint8)


def _soft_shadow(out, sh, k):
    """网点阴影的连续调版本：阴影区域（软边）整体压暗 k。"""
    m = cv2.GaussianBlur(sh.astype(np.float32), (0, 0), 2.0 * R)[..., None]
    return out * (1 - k * m)


def tone_color(img, seed, smooth=False):
    """全彩漫画格（用户 10-19："黑白漫画少用，多用全彩的"；"漫画风格的处理导致人物轮廓和细节都糊了"）：
    全分辨率、保留原画所有细节 —— 原色轻调（对比 × 1.06、彩度 × 1.15）+ 锐化 + 原画描线压深 + 只在最暗 22% 里叠一层 45° 网点。"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = np.clip((lab[..., 0] - 128) * 1.06 + 128, 0, 255)
    lab[..., 1:] = np.clip(128 + (lab[..., 1:] - 128) * 1.15, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    out = np.clip(out * 1.4 - cv2.GaussianBlur(out, (0, 0), 1.1) * 0.4, 0, 255)
    g = gray(img)
    if img.shape[:2] == (H, W):
        sh = g < np.percentile(g[::4, ::4], 22)
        if smooth:
            out = _soft_shadow(out, sh, 0.22)
        else:
            ht = halftone_c(np.clip(g.astype(np.float32) * 2.2, 0, 255).astype(np.uint8), 6 * R, 45) == 0
            out = np.where((sh & ht)[..., None], out * 0.62, out)
    L_ = lineart(img, thick=1)[..., None]
    out = out * (1 - 0.8 * L_) + 14 * 0.8 * L_
    return out.astype(np.uint8)


def tone_mono(img, seed, acc, smooth=False):
    """单色漫画格（用户 10-19："漫画的全彩可以改为 monocolor，多用单色，黑白占比少一点"）：
    全分辨率灰阶（色阶拉开 + 锐化，细节不糊）→ 映射到人物配色的双色调（暗部 = 角色暗色压深，亮部 = 角色亮色混纸色）；
    最暗 22% 叠 45° 网点、原画描线压成墨色。"""
    g = gray(img).astype(np.float32)
    lo, hi = np.percentile(g[::4, ::4], [2, 98])
    g = np.clip((g - lo) / max(1.0, hi - lo), 0, 1)
    g = np.clip(g * 1.35 - cv2.GaussianBlur(g, (0, 0), 1.1) * 0.35, 0, 1)
    acc = acc or {"dk": (40, 30, 60), "hi": (230, 220, 240), "main": (120, 100, 200)}
    dark = np.array(acc["dk"], np.float32) * 0.45
    light = np.array(acc["hi"], np.float32) * 0.45 + np.array((248, 244, 236), np.float32) * 0.55
    mid = np.array(acc["main"], np.float32) * 0.85
    t = g[..., None]
    out = np.where(t < 0.5, dark + (mid - dark) * (t / 0.5), mid + (light - mid) * ((t - 0.5) / 0.5))
    if img.shape[:2] == (H, W):
        gs = (g * 255).astype(np.uint8)
        sh = gs < np.percentile(gs[::4, ::4], 22)
        if smooth:
            out = _soft_shadow(out, sh, 0.24)
        else:
            ht = halftone_c(np.clip(gs.astype(np.float32) * 2.2, 0, 255).astype(np.uint8), 6 * R, 45) == 0
            out = np.where((sh & ht)[..., None], out * 0.6, out)
    L_ = lineart(img, thick=1)[..., None]
    out = out * (1 - 0.85 * L_) + dark * 0.5 * 0.85 * L_
    return np.clip(out, 0, 255).astype(np.uint8)


def tone_duo(img, seed, ca, cb, smooth=False):
    """双人对打的漫画格（用户 10-19："双人对打的漫画风格化处理可以改成双色"）：两色 riso 套印——
    A 色印暗部（实地 + 原画线），B 色印中间调（15° 网点）；纸色做高光。全分辨率，细节不糊。"""
    g = gray(img).astype(np.float32)
    lo, hi = np.percentile(g[::4, ::4], [2, 98])
    g = np.clip((g - lo) / max(1.0, hi - lo), 0, 1)
    g = np.clip(g * 1.35 - cv2.GaussianBlur(g, (0, 0), 1.1) * 0.35, 0, 1)
    ca = np.array(ca, np.float32) * 0.62; cb = np.array(cb, np.float32)
    paper = np.array((246, 241, 230), np.float32)
    a1 = np.clip((0.55 - g) / 0.55, 0, 1) ** 0.9
    a2 = np.clip(1 - np.abs(g - 0.50) / 0.42, 0, 1) * 0.9
    if img.shape[:2] == (H, W) and not smooth:
        dots = halftone_c(np.clip((1 - a2) * 255, 0, 255).astype(np.uint8), 5 * R, 15) == 0
        a2 = a2 * 0.45 + dots.astype(np.float32) * 0.55 * (a2 > 0.08)
    out = paper * (1 - a1[..., None] * (1 - ca / 255)) * (1 - a2[..., None] * (1 - cb / 255))
    L_ = lineart(img, thick=1)[..., None]
    out = out * (1 - 0.85 * L_) + ca * 0.4 * 0.85 * L_
    return np.clip(out, 0, 255).astype(np.uint8)


def tone_ink(img, seed, smooth=False):
    """动作格（黑白）：全分辨率轻高斯（不再半分辨率双边，细节不糊）→ 灰阶分位数 → 阴影网点 + 描线。"""
    h, w = img.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0, 0), 0.7)
    q12, q75, q96 = np.percentile(g, [12, 75, 96])
    tone = 0.18 + 0.82 * np.clip((g.astype(np.float32) - q12) / max(1, q75 - q12), 0, 1)
    if smooth:                                              # 连续调墨稿：S 曲线压成高反差，不出网点
        u = np.clip((tone - 0.25) / 0.6, 0, 1)
        bw = (12 + 233 * u * u * (3 - 2 * u)).astype(np.float32)
    else:
        if (h, w) == (H, W):
            ht = halftone_c((tone * 255).astype(np.uint8), 7 * R, 45)
        else:
            ht = SF.halftone((tone * 255).astype(np.uint8), 7 * R)
        bw = np.where(ht > 0, 245, 12).astype(np.float32)
    bw[g < q12] = 12
    bw[g > q96] = 248
    L = lineart(img, thick=1)
    bw = bw * (1 - L) + 12 * L
    return bw.astype(np.uint8)


def poly_mask(poly, shape=(H, W)):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 255, cv2.LINE_AA)
    return m


def transform_poly(poly, s, dx, dy, rot_deg):
    cx = sum(p[0] for p in poly) / len(poly); cy = sum(p[1] for p in poly) / len(poly)
    a = math.radians(rot_deg); ca, sa = math.cos(a), math.sin(a)
    out = []
    for x, y in poly:
        x0, y0 = (x - cx) * s, (y - cy) * s
        out.append((cx + x0 * ca - y0 * sa + dx, cy + x0 * sa + y0 * ca + dy))
    return out


def layout(kind, face=None):
    """返回格列表：poly、取景（源图归一化中心 / 宽度）、格内处理、入场方向。"""
    G = 26 * R
    fx, fy = (face if face is not None else (0.5, 0.42))
    if kind == "band":
        y0, y1 = H * 0.30, H * 0.70
        return [dict(poly=[(W * 0.07, y0), (W * 0.93, y0), (W * 0.93, y1), (W * 0.07, y1)], view=(0.5, fy, 1.0), dir=(0, 1))]
    if kind == "impact":
        mx, my = 70 * R, 60 * R
        xs, sl, ym = W * 0.60, 150 * R, H * 0.47
        xm = lambda y: xs + sl - (2 * sl) * (y - my) / (H - 2 * my)
        return [dict(poly=[(mx, my), (xs + sl, my), (xs - sl, H - my), (mx, H - my)], view=(0.5, 0.5, 1.0), dir=(-1, 0), lines=True),
                dict(poly=[(xs + sl + G, my), (W - mx, my), (W - mx, ym), (xm(ym) + G, ym)], view=(fx, fy, 0.30), dir=(1, -1)),
                dict(poly=[(xm(ym + G) + G, ym + G), (W - mx, ym + G), (W - mx, H - my), (xs - sl + G, H - my)], view=(0.5, 0.62, 0.45),
                     dir=(1, 1), ink=True)]
    if kind == "story":
        m = 56 * R
        top = [(m, m), (W - m, m), (W - m, H * 0.50), (m, H * 0.50 + 40 * R)]
        yb = H * 0.50 + 40 * R + G
        w3 = (W - 2 * m - 2 * G) / 3
        bl = [(m + k * (w3 + G), yb - (40 * R if k == 0 else 0)) for k in range(3)]
        panels = [dict(poly=top, view=(0.5, 0.45, 1.0), dir=(0, -1), freeze=True)]
        for k in (2, 1, 0):                                     # 右 → 左
            x0 = m + k * (w3 + G)
            yt = H * 0.50 + G + (40 * R * (1 - k / 2.0))
            poly = [(x0, yt), (x0 + w3, yt - 40 * R * 0.5), (x0 + w3, H - m), (x0, H - m)]
            view = [(fx, fy, 0.34), (0.5, 0.55, 0.6), (0.5, 0.5, 1.0)][2 - k]
            panels.append(dict(poly=poly, view=view, dir=(1, 0), freeze=(k != 0)))
        return panels
    # slant
    m = 50 * R
    return [dict(poly=[(m, m), (W - m, m), (W - m, H * 0.40), (m, H * 0.62)], view=(0.5, 0.42, 1.0), dir=(-1, -1)),
            dict(poly=[(m, H * 0.62 + 30 * R), (W - m, H * 0.40 + 30 * R), (W - m, H - m), (m, H - m)], view=(fx, fy, 0.42), dir=(1, 1))]


class MangaPage:
    def __init__(self, kind, seed=0, face=None, tone="gray", acc=None, caption=None, num=7, tag="", smooth=False):
        self.kind, self.seed, self.tone = kind, seed, tone
        self.smooth = smooth
        self.panels = layout(kind, face)
        self.acc = acc
        self.caption = caption
        self.num, self.tag = num, tag
        self.holes = [p["poly"] for p in self.panels]
        self.frozen = {}

    def _fill(self, img, p, k_push, seed):
        """把源图按 view 取景、覆盖到格的外接框里（cover），再按格内处理。"""
        xs = [q[0] for q in p["poly"]]; ys = [q[1] for q in p["poly"]]
        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
        bw, bh = bx1 - bx0, by1 - by0
        vx, vy, vw = p["view"]
        sw = W * vw / k_push
        sw = max(sw, bw / k_push)                            # 不放大取景（用户："漫画风格下的人物细节以及场景细节不能糊"）：最多只有缓推的 1.06 倍
        sh = sw * bh / bw
        if sh > H:
            sh = H; sw = sh * bw / bh
        sx0 = float(np.clip(vx * W - sw / 2, 0, W - sw)); sy0 = float(np.clip(vy * H - sh / 2, 0, H - sh))
        M = np.float32([[bw / sw, 0, bx0 - sx0 * bw / sw], [0, bh / sh, by0 - sy0 * bh / sh]])
        up = bw / sw > 1.0
        crop = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC if up else cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
        if up:                                               # 缓推带来的轻微放大：补一点锐度
            crop = np.clip(crop.astype(np.float32) * 1.25 - cv2.GaussianBlur(crop, (0, 0), 1.0).astype(np.float32) * 0.25, 0, 255).astype(np.uint8)
        if self.tone == "color":                              # 全彩页：每格都保留原色细节（强调格也不再做双色）
            return tone_color(crop, seed, self.smooth)
        if self.tone == "mono":                               # 单色页（人物配色双色调）
            return tone_mono(crop, seed, self.acc, self.smooth)
        if self.tone == "duo" and getattr(self, "duo", None):  # 双人对打：两人各一色
            return tone_duo(crop, seed, self.duo[0], self.duo[1], self.smooth)
        if p.get("ink"):
            t = tone_gray(crop, seed)
            col = np.array(self.acc["main"] if self.acc else (226, 40, 40), np.float32)
            th = float(np.clip(np.percentile(t, 55), 60, 190))   # 暗源（夜景 / 竹林）按中位数切，不然整格都是强调色
            return np.where((t > th)[..., None], np.array((246, 240, 234), np.float32), col).astype(np.uint8)
        if self.tone == "ink" and p is self.panels[0]:          # 只有主格走描线网点；特写 / 细节格保留灰阶细节
            g = tone_ink(crop, seed, self.smooth)
        else:
            g = tone_gray(crop, seed)
        return np.dstack([g] * 3)

    def frame(self, f, img, snap=None):
        """f = 镜头内帧号；img = 当前源帧；snap(k) = 取第 k 格定格用的源帧（story 版式），None 就用 img。"""
        page = np.dstack([paper_tone(self.kind, None, self.smooth)] * 3).astype(np.uint8).copy()
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        Bw = max(3, int(6 * R))
        for k, p in enumerate(self.panels):
            fk = 2 + 4 * k
            if f < fk:
                continue
            u = (f - fk) / 6.0
            s = 1.0 + 0.12 * (1 - bo(min(1.0, u), 2.0)) if u < 1 else 1.0
            dx, dy = p["dir"][0] * 60 * R * (1 - eo(u)), p["dir"][1] * 60 * R * (1 - eo(u))
            rot = 1.5 * (1 - eo(u)) * (1 if k % 2 else -1)
            poly = transform_poly(p["poly"], s, dx, dy, rot)
            src = img
            if p.get("freeze") and snap is not None and k < len(self.panels) - 1:
                src = self.frozen.get(k)
                if src is None:
                    src = snap(fk)
                    self.frozen[k] = src
            push = 1.0 + 0.06 * cl(max(0, f - fk) / 90.0)
            cont = self._fill(src, p, push, self.seed * 31 + k * 7 + (f // 2) % 3)
            if p.get("lines") and self.tone == "ink":
                cont = speed_lines(cont, (W * 0.34, H * 0.48), self.seed * 13 + f // 2)
            m = poly_mask(poly)[..., None].astype(np.float32) / 255
            page = (page * (1 - m) + cont * m).astype(np.uint8)
            d.line(poly + [poly[0]], fill=(10, 10, 12, 255), width=Bw, joint="curve")
        if self.caption and f >= 8:
            caption_box(d, self.caption, f - 8, self.kind)
        if f >= 4:
            d.text((70 * R, H - 52 * R), "— %02d —" % self.num, font=font(SF.F_BODONI_BLK, 26 * R), fill=(10, 10, 12, 255))
            if self.tag:
                SF.spaced(d, (W - 70 * R - 360 * R, H - 46 * R), self.tag, font(SF.F_BAHN, 18 * R), (10, 10, 12, 255), 6 * R)
        b = Image.fromarray(page).convert("RGBA"); b.alpha_composite(lay)
        return np.asarray(b.convert("RGB"))


def speed_lines(img, c, seed, n=200):
    rng = np.random.default_rng(seed)
    lay = Image.new("L", (W, H), 0); ds = ImageDraw.Draw(lay)
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi); r0 = rng.uniform(0.34, 0.52) * H; r1 = 1.4 * W
        w0 = rng.uniform(2, 6) * R
        p1 = (c[0] + r0 * math.cos(a), c[1] + r0 * math.sin(a))
        pa = (c[0] + r1 * math.cos(a - w0 / r1 * 40), c[1] + r1 * math.sin(a - w0 / r1 * 40))
        pb = (c[0] + r1 * math.cos(a + w0 / r1 * 40), c[1] + r1 * math.sin(a + w0 / r1 * 40))
        ds.polygon([p1, pa, pb], fill=150)
    m = np.asarray(lay, np.float32)[..., None] / 255
    return (img * (1 - m) + 250 * m).astype(np.uint8)


def caption_box(d, text, f, kind):
    """竖排说明框（明朝 Bold，白底细框），逐字落下。text：日文一列 ≤ 9 字，多列自动换。"""
    cols = [text[i:i + 8] for i in range(0, len(text), 8)][:3]
    fj = SF.vfont(SF.F_SERIF_CJK, 46 * R, "Bold")
    lh, cw = 54 * R, 64 * R
    bw_, bh_ = len(cols) * cw + 36 * R, max(len(c) for c in cols) * lh + 48 * R
    if kind == "band":
        bx, by = W * 0.93 - bw_ + 20 * R, H * 0.70 + 26 * R if H * 0.70 + 26 * R + bh_ < H - 30 * R else 40 * R
        bx, by = W - 90 * R - bw_, 60 * R
    elif kind == "impact":
        bx, by = W * 0.60 - 150 * R * 0.1 - bw_ - 60 * R, 116 * R
    else:
        bx, by = W - 90 * R - bw_, 80 * R
    k = eo(f / 6.0)
    d.rectangle([bx, by, bx + bw_, by + bh_ * k], fill=(250, 250, 248, 255), outline=(10, 10, 12, 255), width=max(2, int(3 * R)))
    n = 0
    for c_i, col in enumerate(cols):
        x0 = bx + bw_ - 18 * R - (c_i + 1) * cw + 8 * R
        for j, ch in enumerate(col):
            if f >= 3 + n * 1.5:
                d.text((x0, by + 24 * R + j * lh), ch, font=fj, fill=(10, 10, 12, 255))
            n += 1


# ---------------------------------------------------------------- B riso 角色海报（活画面）
RISO_RED, RISO_INK, RISO_PAPER = (232, 62, 48), (24, 30, 52), (242, 236, 224)


class RisoPoster:
    def __init__(self, big, sub, jp, num="01", red=None, smooth=False):
        self.big, self.sub, self.jp, self.num = big, sub, jp, num
        self.smooth = smooth                                # 电影质感段：印版不出网点（连续调），颗粒交给 v7_cine
        self.red = red or RISO_RED
        fb = font(SF.F_BODONI_BLK, 250 * R if len(big) <= 8 else 190 * R)
        self.t_ink = SF.squeeze_text(big, fb, RISO_INK, sx=0.78, track=-6 * R)
        self.t_red = SF.squeeze_text(big, fb, self.red, sx=0.78, track=-6 * R)
        self.cache = {}

    def plates(self, img, f):
        key = int(f) // 5                                   # 12 fps 步进
        if key not in self.cache:
            g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray(img)).astype(np.float32) / 255
            g = cv2.GaussianBlur(g, (0, 0), 0.8)
            red_d = np.clip((1 - g) * 0.95, 0, 1); ink_d = np.clip((0.42 - g) / 0.42, 0, 1) ** 1.1
            if self.smooth:
                scr = lambda dens, ang: cv2.GaussianBlur(dens.astype(np.float32), (0, 0), 0.8)
            else:
                scr = lambda dens, ang: cv2.GaussianBlur((halftone_c((255 * (1 - dens)).astype(np.uint8), 6 * R, ang) == 0).astype(np.float32), (0, 0), 0.6)
            self.cache = {key: (scr(red_d, 15), np.maximum(scr(ink_d, 45), lineart(img)))}
        return self.cache[key]

    def frame(self, f, img):
        red, ink = self.plates(img, f)
        k = bo(f / 12.0) if f >= 0 else 0.0
        sh = lambda a, dx, dy: cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderValue=0)
        red = sh(red, -28 * R * (1 - k), 18 * R * (1 - k)); ink = sh(ink, 3 * R + 40 * R * (1 - k), 2 * R - 26 * R * (1 - k))
        kr = 1 - np.array(self.red, np.float32) / 255; ki = 0.92 * (1 - np.array(RISO_INK, np.float32) / 255)
        out = np.array(RISO_PAPER, np.float32) * (1 - red[..., None] * kr)
        out = out * (1 - ink[..., None] * ki) + (0.0 if self.smooth else grain((H, W), int(f) // 2, 6.0)[..., None])
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        M = 144; INKc = RISO_INK + (255,); REDc = tuple(self.red) + (255,)
        by = int(960 - self.t_ink.height * 0.80)
        rise = lambda im, v: (lambda L_: (L_.alpha_composite(im, (0, int((1 - v) * im.height))), L_)[1])(Image.new("RGBA", im.size, (0, 0, 0, 0)))
        if f >= 3:
            lay.alpha_composite(rise(self.t_red, eo((f - 5) / 7.0)), (M + int(5 * R + 24 * R * (1 - k)), by + int(4 * R)))
            lay.alpha_composite(rise(self.t_ink, eo((f - 3) / 7.0)), (M, by))
        if f >= 8:
            SF.spaced(d, (M + 4, by - 40 * R), ("CLASS  —  No.%s" % self.num)[:int((f - 8) * 2)], font(SF.F_BAHN, 26 * R), INKc, 9 * R)
        ry = 960 + 26 * R
        rw = self.t_ink.width * eo((f - 9) / 8.0)
        if rw > 1:
            d.line([M, ry, M + rw, ry], fill=INKc, width=max(1, int(1.5 * R)))
        if f >= 11:
            SF.spaced(d, (M + 2, ry + 14 * R), self.sub.upper()[:int((f - 11) * 2)], font(SF.F_BAHN, 34 * R), INKc, 10 * R)
        fj = SF.vfont(SF.F_SERIF_CJK, 40 * R, "Bold")
        x0, y0 = W - M - 40 * R, 220
        for kk, ch in enumerate(self.jp):
            fk = 10 + 2 * kk
            if f >= fk:
                d.text((x0, y0 + kk * 46 * R - 10 * R * (1 - eo((f - fk) / 3.0))), ch, font=fj, fill=REDc)
        L_ = 34 * R * eo(f / 6.0)
        if L_ > 1:
            for (x, y, sx, sy) in ((M * 0.5, 72, -1, -1), (W - M * 0.5, 72, 1, -1), (M * 0.5, H - 72, -1, 1), (W - M * 0.5, H - 72, 1, 1)):
                d.line([x + sx * 8, y, x + sx * (8 + L_), y], fill=INKc, width=max(1, int(1.2 * R)))
                d.line([x, y + sy * 8, x, y + sy * (8 + L_)], fill=INKc, width=max(1, int(1.2 * R)))
        if f >= 6:
            d.text((M, 96), "PROVANT // HEROIC SPIRIT FILE", font=font(SF.F_MONO, 15 * R), fill=INKc)
        b = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA"); b.alpha_composite(lay)
        o = np.asarray(b.convert("RGB"))
        s_ = 1.0 + 0.03 * cl(f / 80.0)
        return cv2.warpAffine(o, np.float32([[s_, 0, W / 2 * (1 - s_)], [0, s_, H / 2 * (1 - s_)]]), (W, H), borderMode=cv2.BORDER_REFLECT)

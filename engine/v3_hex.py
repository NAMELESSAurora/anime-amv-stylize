# -*- coding: utf-8 -*-
"""
EVA 式六边形（用户 10-19 第 5 条："六边形蜂巢阵可以多用一点，可以参考 EVA 的风格"）。
  hex_field  蜂巢面板：平顶六边形阵（不规则缺格），从起点按距离波形逐格弹出（0.4→1 back-out，5 帧）；
             部分格实心强调色 + 黑字（承認 / 否決 / 警告 / 待機 / MAGI / 601 …，压缩明朝 = EVA 字体感），
             其余细框 + 小编号；每拍翻一格（≤ 3 次 / 秒）；顶部切角标题条（"MAGI SYSTEM // 承認"）。
  hex_wipe   蜂巢转场：全屏六边形格按离动机点的距离（加抖动）依次翻转 A→B，翻转中的格先亮强调色 3 帧、描边。
画在 hud_v2.Canvas 上（预乘 alpha 层 + 文字层）。
"""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw

import style_frames as SF
from style_frames import W, H, R, font
from cards import eo, bo, cl
import hud_elems as HE

WHITE = (236, 242, 244)
WORDS = ["承認", "否決", "警告", "待機", "MAGI", "601", "BLUE", "照合", "同期", "解除", "封印", "起動"]
_TXT = {}


def cond_text(s, size, col):
    """压缩明朝（EVA 标题字的感觉）：Noto Serif Black 横向压到 0.62。"""
    key = (s, int(size), tuple(col))
    if key not in _TXT:
        f = SF.vfont(SF.F_SERIF_CJK, size, "Black") if any(ord(c) > 0x3000 for c in s) else SF.font(SF.F_BODONI_BLK, size)
        _TXT[key] = SF.squeeze_text(s, f, col, sx=0.62)
    return _TXT[key]


def hex_pts(c, r, scale=1.0):
    return [(c[0] + r * scale * math.cos(math.radians(60 * k)), c[1] + r * scale * math.sin(math.radians(60 * k))) for k in range(6)]


def hex_field(cv, origin, f, bk, bf, acc, seed, cols=7, rows=4, r=None, title=True):
    r = r or 40 * R
    rng = np.random.default_rng(seed)
    dx, dy = 1.5 * r, math.sqrt(3) * r
    cells = []
    for q in range(cols):
        for k in range(rows):
            if rng.random() < 0.74 or (q, k) == (0, 0):
                x = origin[0] + q * dx
                y = origin[1] + k * dy + (q % 2) * dy / 2
                cells.append((q, k, x, y))
    if not cells:
        return
    filled = set(int(i) for i in rng.choice(len(cells), max(1, len(cells) // 5), replace=False))
    blink = int(bk + seed) % len(cells)
    for idx, (q, k, x, y) in enumerate(cells):
        d = math.hypot(q, k * 1.15)
        fk = 2 + d * 1.6
        if f < fk:
            continue
        sc = 0.4 + 0.6 * bo(min(1.0, (f - fk) / 5.0), 2.0)
        pts = hex_pts((x, y), r * 0.93, sc)
        on = (idx in filled) != (idx == blink and bf < 30)
        if on:
            HE.fpoly(cv, pts, acc["main"], 0.92)
            if sc > 0.9:
                w = WORDS[(idx + seed) % len(WORDS)]
                im = cond_text(w, 26 * R if len(w) <= 2 else 20 * R, (12, 10, 10))
                cv.txt.alpha_composite(im, (int(x - im.width / 2), int(y - im.height / 2)))
        else:
            HE.poly(cv, pts, WHITE, 0.75, 1.3 * R)
            HE.poly(cv, hex_pts((x, y), r * 0.70, sc), acc["hi"], 0.35, 1)
            if sc > 0.9 and (idx * 7 + seed) % 3 == 0:
                cv.text((x - 12 * R, y - 7 * R), "%03d" % ((idx * 37 + seed) % 1000), 9 * R, WHITE, 0.8)
    if title and f >= 4:
        x0, y0 = origin[0] - r, origin[1] - r * 1.9
        tw = 330 * R * eo((f - 4) / 8.0)
        if tw > 4:
            HE.fpoly(cv, HE.chamfer_rect(x0, y0, x0 + tw, y0 + 30 * R, 12 * R, (1, 0, 1, 0)), (14, 12, 12), 0.85)
            HE.poly(cv, HE.chamfer_rect(x0, y0, x0 + tw, y0 + 30 * R, 12 * R, (1, 0, 1, 0)), acc["main"], 1.0, 1.5 * R)
            if tw > 200 * R:
                im = cond_text("MAGI SYSTEM", 24 * R, acc["main"])
                cv.txt.alpha_composite(im, (int(x0 + 16 * R), int(y0 + 2 * R)))
                im2 = cond_text(WORDS[(bk + seed) % 4], 26 * R, WHITE)
                cv.txt.alpha_composite(im2, (int(x0 + 16 * R + im.width + 18 * R), int(y0 + 1 * R)))


# ---------------------------------------------------------------- 蜂巢转场
_ID = {}


def _hex_ids(r):
    """每个像素所在的平顶六边形格（轴坐标取整），缓存。返回 (id 图, 格心 x, 格心 y)。"""
    if r in _ID:
        return _ID[r]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    q = (2.0 / 3.0 * xx) / r
    rr = (-1.0 / 3.0 * xx + math.sqrt(3) / 3.0 * yy) / r
    x, z = q, rr; y_ = -x - z
    rx, ry, rz = np.round(x), np.round(y_), np.round(z)
    dx, dy, dz = np.abs(rx - x), np.abs(ry - y_), np.abs(rz - z)
    m1 = (dx > dy) & (dx > dz); m2 = (~m1) & (dy > dz)
    rx = np.where(m1, -ry - rz, rx)
    rz = np.where(~m1 & ~m2, -rx - ry, rz)
    qi, ri = rx.astype(np.int32), rz.astype(np.int32)
    cid = (qi + 1000) * 4000 + (ri + 1000)
    cx = r * 1.5 * qi
    cy = r * math.sqrt(3) * (ri + qi / 2.0)
    _ID[r] = (cid, cx.astype(np.float32), cy.astype(np.float32))
    return _ID[r]


def hex_wipe(A, B, f, n, origin, acc, r=None, seed=5):
    r = int(r or 64 * R)
    cid, cx, cy = _hex_ids(r)
    d = np.hypot(cx - origin[0], cy - origin[1])
    jit = ((cid.astype(np.int64) * 2654435761) % 1000).astype(np.float32) / 1000.0
    dn = d / max(1.0, d.max())
    t0 = (dn * 0.78 + jit * 0.12) * (n - 4)            # 每格开始翻的帧
    k = f - t0
    flip = (k >= 0) & (k < 3)
    done = k >= 3
    out = np.where(done[..., None], B, A).astype(np.float32)
    col = np.array(acc["main"], np.float32) * 0.85
    out = np.where(flip[..., None], out * 0.35 + col * 0.65, out)
    edge = _hex_edges(r)                                 # 正在翻 / 刚翻完的格描边（亮强调色，6 帧内淡掉）
    lit = edge & (k >= 0) & (k < 7)
    a = np.clip(1 - (k - 3) / 4.0, 0, 1) * lit
    hi = np.array(acc["hi"], np.float32)
    out = out * (1 - a[..., None]) + hi * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


_ED = {}


def _hex_edges(r):
    if r not in _ED:
        cid = _hex_ids(r)[0]
        e = np.zeros(cid.shape, bool)
        e[:, 1:] |= cid[:, 1:] != cid[:, :-1]
        e[1:, :] |= cid[1:, :] != cid[:-1, :]
        _ED[r] = cv2.dilate(e.astype(np.uint8), np.ones((max(2, int(2 * R)),) * 2, np.uint8)) > 0
    return _ED[r]

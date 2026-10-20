# -*- coding: utf-8 -*-
"""
HUD 元素库 2（用户 10-18："UI 除了圆、矩、三角外多设计点别的，不然容易视觉疲劳"）。
每个镜头按种子从库里抽 3–4 种，取景框 / 采样窗也有 3 种外形轮换，相邻镜头的 UI 语汇不重复。
  magic   Fate 魔法阵：双环 + 程序生成的符文环 + 内接六芒星（一只三角用强调色，按拍转 15°）+ 顶点小环
  radar   雷达扫描：虚线圆 + 渐隐扇形扫掠 + 扫到才亮的菱形回波点
  hexes   蜂巢残片：六边形格，几格强调色填充 / 斜纹，按拍翻一格
  matrix  矩阵面板：十六进制值格（行列表头、强调色高亮簇）/ 0-1 二进制流（用户加的："可以试试加点矩阵"）
  chevron 箭头串（>>>）行进高亮
  code    条码 + 数据矩阵（每 6 帧刷新）
  hazard  斜纹警示条（切角两端）
  plus    十字 / 叉 / 菱形点阵（替代圆点网格）
  orbit   斜透视轨道椭圆（虚线）+ 卫星点
  graph   折线图面板（RMS 历史，菱形节点 + 网格 + 轴标）
（立方体用户不要，已删；雷达降权。）
画法同 hud_v2.Canvas（OpenCV 抗锯齿、预乘 alpha）。
"""
import math
import numpy as np
import cv2

from style_frames import W, H, R
from cards import eo, cl

WHITE = (236, 242, 244)
GREY = (150, 158, 164)
DGREY = (92, 98, 104)
SH = 4
SC = 1 << SH


def _p(x, y):
    return (int(round(x * SC)), int(round(y * SC)))


def poly(cv, pts, col, a=1.0, th=1.0, closed=True):
    arr = np.array([_p(*p) for p in pts], np.int32)
    cv2.polylines(cv.a, [arr], closed, (col[0] * a, col[1] * a, col[2] * a, 255 * a), max(1, int(round(th))), cv2.LINE_AA, SH)


def fpoly(cv, pts, col, a=1.0):
    arr = np.array([_p(*p) for p in pts], np.int32)
    cv2.fillPoly(cv.a, [arr], (col[0] * a, col[1] * a, col[2] * a, 255 * a), cv2.LINE_AA, SH)


def diamond(cv, c, r, col, a=1.0, fill=False):
    pts = [(c[0], c[1] - r), (c[0] + r, c[1]), (c[0], c[1] + r), (c[0] - r, c[1])]
    (fpoly(cv, pts, col, a) if fill else poly(cv, pts, col, a, 1))


def hexagon(c, r, rot=0.0):
    return [(c[0] + r * math.cos(math.radians(60 * k + rot)), c[1] + r * math.sin(math.radians(60 * k + rot))) for k in range(6)]


def chamfer_rect(x0, y0, x1, y1, ch, corners=(1, 1, 1, 1)):
    """切角矩形（八边形外形）。corners = (左上, 右上, 右下, 左下) 哪几个角切。"""
    c = [ch * k for k in corners]
    return [(x0 + c[0], y0), (x1 - c[1], y0), (x1, y0 + c[1]), (x1, y1 - c[2]), (x1 - c[2], y1), (x0 + c[3], y1), (x0, y1 - c[3]), (x0, y0 + c[0])]


# ---------------------------------------------------------------- 元素
def magic(cv, c, r, f, bk, bf, acc, seed):
    step = 15 * (bk + eo(bf / 6.0))
    cv.circle(c, r, WHITE, 0.85, 1.6 * R); cv.circle(c, r * 0.90, WHITE, 0.6, 1)
    cv.circle(c, r * 0.62, WHITE, 0.55, 1)
    rng = np.random.default_rng(seed)
    n = 44                                                  # 符文环：每个字 2–4 笔短划
    for k in range(n):
        a = math.radians(k * 360.0 / n - step * 0.5)
        rr = r * 0.95
        gx, gy = c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)
        u = (math.cos(a + math.pi / 2), math.sin(a + math.pi / 2)); v = (math.cos(a), math.sin(a))
        s = r * 0.022
        for _ in range(int(rng.integers(2, 5))):
            p0 = rng.uniform(-1, 1, 2); p1 = p0 + rng.choice([-1, 1]) * np.array([1.2, 0]) if rng.random() < 0.5 else p0 + np.array([0, rng.choice([-1, 1]) * 1.2])
            q0 = (gx + s * (p0[0] * u[0] + p0[1] * v[0]), gy + s * (p0[0] * u[1] + p0[1] * v[1]))
            q1 = (gx + s * (p1[0] * u[0] + p1[1] * v[0]), gy + s * (p1[0] * u[1] + p1[1] * v[1]))
            cv.line(q0, q1, WHITE, 0.8, 1)
    for tri, col, al in ((0, acc["main"], 1.0), (60, WHITE, 0.75)):   # 六芒星
        pts = [(c[0] + r * 0.88 * math.cos(math.radians(tri + 120 * k + step)), c[1] + r * 0.88 * math.sin(math.radians(tri + 120 * k + step)))
               for k in range(3)]
        poly(cv, pts, col, al, 1.4 * R if col is not WHITE else 1)
        for p in pts:
            cv.circle(p, r * 0.06, col, al, 1)
    poly(cv, hexagon(c, r * 0.44, step * -1), WHITE, 0.6, 1)


def radar(cv, c, r, f, acc, seed):
    for k in range(0, 360, 8):
        cv.arc(c, r, k, k + 5, WHITE, 0.6, 1)
    cv.circle(c, r * 0.5, WHITE, 0.35, 1)
    for a in (0, 90):
        cv.line((c[0] - r, c[1]) if a == 0 else (c[0], c[1] - r), (c[0] + r, c[1]) if a == 0 else (c[0], c[1] + r), WHITE, 0.35, 1)
    th = (f * 6.0) % 360
    for k in range(22):                                     # 扇形扫掠：渐隐的放射线
        a = math.radians(th - k * 1.6)
        cv.line(c, (c[0] + r * math.cos(a), c[1] + r * math.sin(a)), acc["hi"], 0.55 * (1 - k / 22.0), 1.4 * R if k == 0 else 1)
    rng = np.random.default_rng(seed)
    for _ in range(6):                                      # 回波：扫过后亮、再慢慢暗
        ang = rng.uniform(0, 360); rr = r * rng.uniform(0.25, 0.9)
        age = ((th - ang) % 360) / 360.0
        if age < 0.6:
            p = (c[0] + rr * math.cos(math.radians(ang)), c[1] + rr * math.sin(math.radians(ang)))
            diamond(cv, p, 5 * R, acc["main"], 1 - age / 0.6, fill=True)


def hexes(cv, o, cell, f, bk, acc, seed):
    rng = np.random.default_rng(seed)
    cells = []
    for q in range(5):
        for rr in range(4):
            x = o[0] + q * cell * 1.5; y = o[1] + rr * cell * math.sqrt(3) + (q % 2) * cell * math.sqrt(3) / 2
            if rng.random() < 0.62:
                cells.append((x, y))
    flip = bk % max(1, len(cells))
    for k, (x, y) in enumerate(cells):
        pts = hexagon((x, y), cell * 0.94)
        if k in (1, 4) or k == flip:
            fpoly(cv, pts, acc["main"], 0.9)
        elif k % 5 == 2:
            for j in range(-4, 5):                          # 斜纹格
                cv.line((x + j * cell * 0.22 - cell * 0.5, y - cell * 0.6), (x + j * cell * 0.22 + cell * 0.5, y + cell * 0.6), WHITE, 0.5, 1)
            poly(cv, pts, WHITE, 0.8, 1)
        else:
            poly(cv, pts, WHITE, 0.7, 1)


def cube(cv, c, s, bk, bf, acc):
    ang = math.radians(15 * (bk + eo(bf / 6.0)) + 20)
    pts3 = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]

    def pr(p):
        x, y, z = p
        xr = x * math.cos(ang) - z * math.sin(ang); zr = x * math.sin(ang) + z * math.cos(ang)
        return (c[0] + s * (xr * 0.87), c[1] + s * (y * 0.75 + zr * 0.38))
    P = [pr(p) for p in pts3]
    for i in range(8):
        for j in range(i + 1, 8):
            if sum(a != b for a, b in zip(pts3[i], pts3[j])) == 1:
                cv.line(P[i], P[j], WHITE if (i + j) % 3 else acc["hi"], 0.85, 1.2 * R)
    for k, col in ((0, acc["main"]), (1, WHITE), (2, GREY)):  # 三轴
        e = [0, 0, 0]; e[k] = 1.7
        cv.line(pr((0, 0, 0)), pr(tuple(e)), col, 0.8, 1)


def chevron(cv, o, n, f, acc, size=None, vertical=False):
    size = size or 16 * R
    hi = (f // 3) % n
    for k in range(n):
        x, y = (o[0], o[1] + k * size * 1.1) if vertical else (o[0] + k * size * 1.1, o[1])
        if vertical:
            pts = [(x - size, y), (x, y + size * 0.7), (x + size, y)]
        else:
            pts = [(x, y - size), (x + size * 0.7, y), (x, y + size)]
        poly(cv, pts, acc["main"] if k == hi else WHITE, 1.0 if k == hi else 0.55, 2.2 * R if k == hi else 1.4 * R, closed=False)


def code(cv, o, f, acc, seed):
    rng = np.random.default_rng(seed)
    x = o[0]
    for _ in range(34):                                     # 条码
        w = rng.choice([1, 1, 2, 3]) * R
        if rng.random() < 0.6:
            cv.rect((x, o[1]), (x + w, o[1] + 40 * R), WHITE, 0.85)
        x += w + rng.choice([1, 2]) * R
    rng2 = np.random.default_rng(seed + f // 6)             # 数据矩阵（每 6 帧刷新）
    cs = 5 * R; mx = x + 18 * R
    for i in range(12):
        for j in range(12):
            if i in (0, 11) or j == 0 or rng2.random() < 0.45:
                cv.rect((mx + j * cs, o[1] - 10 * R + i * cs), (mx + j * cs + cs - 1, o[1] - 10 * R + i * cs + cs - 1),
                        acc["main"] if (i == 0 and j % 2 == 0) else WHITE, 0.8)


def hazard(cv, o, w, h, f, acc):
    pts = chamfer_rect(o[0], o[1], o[0] + w, o[1] + h, h * 0.6, (1, 0, 1, 0))
    poly(cv, pts, WHITE, 0.8, 1)
    off = (f * 2 * R) % (h * 1.2)
    k = -2
    while k * h * 1.2 < w + h:
        x = o[0] + k * h * 1.2 + off
        stripe = [(x, o[1] + h), (x + h * 0.6, o[1] + h), (x + h * 1.2, o[1]), (x + h * 0.6, o[1])]
        stripe = [(min(max(px, o[0] + 2), o[0] + w - 2), py) for px, py in stripe]
        fpoly(cv, stripe, acc["main"], 0.9)
        k += 1


def plus(cv, region, f, acc, seed):
    x0, y0, x1, y1 = region
    s = 26 * R
    rng = np.random.default_rng(seed)
    yy = y0 + s
    while yy < y1 - s / 2:
        xx = x0 + s
        while xx < x1 - s / 2:
            u = rng.random()
            m = 4 * R
            if u < 0.08:
                diamond(cv, (xx, yy), m, acc["hi"], 0.8)
            elif u < 0.16:
                cv.line((xx - m, yy - m), (xx + m, yy + m), GREY, 0.6, 1); cv.line((xx - m, yy + m), (xx + m, yy - m), GREY, 0.6, 1)
            else:
                cv.line((xx - m, yy), (xx + m, yy), DGREY, 0.8, 1); cv.line((xx, yy - m), (xx, yy + m), DGREY, 0.8, 1)
            xx += s
        yy += s


def orbit(cv, c, r, f, acc, seed):
    rng = np.random.default_rng(seed)
    for k in range(3):
        tilt = rng.uniform(-35, 35); ax = r * (1 - 0.18 * k); ay = ax * rng.uniform(0.22, 0.42)
        for a0 in range(0, 360, 12):
            cv2.ellipse(cv.a, _p(*c), (int(ax * SC), int(ay * SC)), tilt, a0, a0 + 7, (WHITE[0] * 0.6, WHITE[1] * 0.6, WHITE[2] * 0.6, 153),
                        1, cv2.LINE_AA, SH)
        t = math.radians((f * (2 + k)) % 360)
        px, py = ax * math.cos(t), ay * math.sin(t)
        tr = math.radians(tilt)
        p = (c[0] + px * math.cos(tr) - py * math.sin(tr), c[1] + px * math.sin(tr) + py * math.cos(tr))
        cv.circle(p, 4 * R, acc["main"] if k == 0 else WHITE, 1.0, -1)
    cv.circle(c, 6 * R, WHITE, 0.8, 1)


def graph(cv, o, w, h, rms, i0, acc):
    poly(cv, chamfer_rect(o[0], o[1], o[0] + w, o[1] + h, 12 * R, (1, 0, 1, 0)), WHITE, 0.7, 1)
    for k in range(1, 4):
        cv.line((o[0] + 6 * R, o[1] + h * k / 4), (o[0] + w - 6 * R, o[1] + h * k / 4), DGREY, 0.6, 1)
    n = 24
    pts = []
    for k in range(n):
        v = rms[min(len(rms) - 1, max(0, i0 - (n - k) * 6))]
        pts.append((o[0] + 10 * R + k * (w - 20 * R) / (n - 1), o[1] + h - 8 * R - (h - 16 * R) * min(1.0, v)))
    poly(cv, pts, acc["hi"], 0.95, 1.4 * R, closed=False)
    for k, p in enumerate(pts):
        if k % 4 == 3:
            diamond(cv, p, 4 * R, acc["main"], 1.0, fill=True)
    cv.text((o[0] + 8 * R, o[1] - 16 * R), "SIG.RMS  //  60F", 8.5 * R, GREY)


def matrix(cv, o, f, acc, seed, mode=0):
    """矩阵面板（用户："可以试试加点矩阵"）。mode 0：8×6 两位十六进制值，行列表头，一簇格子强调色高亮，每 4 帧刷新；
    mode 1：0/1 二进制流，列每 3 帧向下滚一格，一列高亮。"""
    if mode == 0:
        cols, rows, cw, chh = 8, 6, 38 * R, 24 * R
        w, h = cols * cw, rows * chh
        HE_poly = poly
        HE_poly(cv, chamfer_rect(o[0] - 6 * R, o[1] - 26 * R, o[0] + w + 6 * R, o[1] + h + 6 * R, 12 * R, (1, 0, 1, 0)), WHITE, 0.7, 1)
        for j in range(cols):
            cv.text((o[0] + j * cw + 8 * R, o[1] - 22 * R), "%02X" % (j * 16), 7.5 * R, GREY)
        rng = np.random.default_rng(seed * 31 + f // 4)
        hx, hy = int(rng.integers(0, cols - 2)), int(rng.integers(0, rows - 1))
        for i in range(rows):
            cv.line((o[0], o[1] + i * chh), (o[0] + w, o[1] + i * chh), DGREY, 0.5, 1)
            for j in range(cols):
                v = int(rng.integers(0, 256))
                x, y = o[0] + j * cw, o[1] + i * chh
                hot = hx <= j <= hx + 2 and hy <= i <= hy + 1
                if hot:
                    cv.rect((x + 1, y + 1), (x + cw - 2, y + chh - 2), acc["main"], 0.95)
                cv.text((x + 7 * R, y + 4 * R), "%02X" % v, 10 * R, acc["txt"] if hot else WHITE, 1.0 if hot else 0.75)
        cv.text((o[0], o[1] + h + 10 * R), "MEM 0x%04X  //  ROW %d COL %d" % ((seed * 977 + f) % 65536, hy, hx), 8 * R, GREY)
    else:
        cols, rows, cw, chh = 16, 10, 16 * R, 18 * R
        w, h = cols * cw, rows * chh
        poly(cv, chamfer_rect(o[0] - 6 * R, o[1] - 6 * R, o[0] + w + 6 * R, o[1] + h + 6 * R, 10 * R, (0, 1, 0, 1)), WHITE, 0.6, 1)
        hc = (f // 6) % cols
        for j in range(cols):
            rng = np.random.default_rng(seed * 101 + j)
            bits = rng.integers(0, 2, rows + 64)
            off = (f // 3 + j * 5) % 64
            for i in range(rows):
                b = bits[(i - off) % (rows + 64)]
                col, al = (acc["main"], 1.0) if j == hc else (WHITE, 0.35 + 0.5 * (i == (off % rows)))
                cv.text((o[0] + j * cw + 3 * R, o[1] + i * chh), str(int(b)), 10 * R, col, al)


BANK = ["magic", "radar", "hexes", "chevron", "code", "hazard", "orbit", "graph", "matrix"]
# 抽签权重（用户：不要立方体；波形图、蜂巢阵可以；加点矩阵；雷达图少用）
WEIGHT = {"magic": 1.0, "radar": 0.3, "hexes": 1.5, "chevron": 0.8, "code": 0.9, "hazard": 0.8, "orbit": 0.9, "graph": 1.5, "matrix": 1.8}
FAV = ("graph", "hexes", "matrix")


def pick(rng, n=3):
    """按权重不放回抽 n 个；保证至少一个来自 波形图 / 蜂巢 / 矩阵。"""
    names = list(BANK)
    w = np.array([WEIGHT[k] for k in names], float)
    out = list(rng.choice(names, n, replace=False, p=w / w.sum()))
    if not any(k in FAV for k in out):
        out[-1] = FAV[int(rng.integers(len(FAV)))]
    return [str(k) for k in out]


def draw(cv, name, pos, f, bk, bf, acc, seed, rms=None, i0=0):
    x, y = pos
    if name == "magic":
        magic(cv, (x, y), 150 * R, f, bk, bf, acc, seed)
    elif name == "radar":
        radar(cv, (x, y), 120 * R, f, acc, seed)
    elif name == "hexes":
        hexes(cv, (x - 120 * R, y - 90 * R), 26 * R, f, bk, acc, seed)
    elif name == "matrix":
        matrix(cv, (x - 150 * R, y - 70 * R), f, acc, seed, mode=seed % 2)
    elif name == "chevron":
        chevron(cv, (x - 80 * R, y), 7, f, acc)
    elif name == "code":
        code(cv, (x - 150 * R, y - 20 * R), f, acc, seed)
    elif name == "hazard":
        hazard(cv, (x - 160 * R, y - 12 * R), 320 * R, 24 * R, f, acc)
    elif name == "orbit":
        orbit(cv, (x, y), 150 * R, f, acc, seed)
    elif name == "graph":
        graph(cv, (x - 150 * R, y - 60 * R), 300 * R, 120 * R, rms, i0, acc)

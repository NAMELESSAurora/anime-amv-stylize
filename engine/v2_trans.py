# -*- coding: utf-8 -*-
"""
v2 转场（用户 10-18："转场可以加一些 splash 和 glitch 效果"；看过预览后"splash 效果做的太粗糙了…
可以设计些别的特效或者 motion 来替换 splash" → splash 撤掉，换成下面的 lock_zoom / panel_slap）。

glitch      切片错位 + RGB 分离 + 宏块位移（datamosh 感）+ A/B 切片交错 + 少量强调色块；
            包络在切点最强，种子每 2 帧换（hold 2，不逐帧乱闪），不改整体亮度（不做白闪）。
slam        M7 砸入：新镜头 1.22→1.0 指数缓出，叠在 glitch 衰减上。
lock_zoom   M4 HUD 锁定接力：角标咬住 A 的目标（back-out），框内用 Bayer 抖点"解码"出 B 的缩略画面，
            随后镜头从这个窗口穿过去（对数空间缩放，慢-快-稳），A 带径向缩放模糊 + 速度峰值处轻微色散；
            窗口细边 / 角标 / 坐标小字用 B 的强调色，跟着窗口一起涨到满屏后出画。
panel_slap  M9 档案拍贴：6 个档案格按 2 帧间隔弹出（0.6→1.06→1），每格是 B 的取景窗（内容带 20R 视差回位），
            墨色细框 + 强调色标题条；随后各格缓动扩到满屏合并成 B。亮度按格子面积阶梯上升（暗→纸灰不跳）。
"""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw

from style_frames import W, H, R, font
import style_frames as SF
from cards import eo, bo, cl

INK = (14, 13, 12)
BAYER8 = np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44, 4, 36, 14, 46, 6, 38],
                   [60, 28, 52, 20, 62, 30, 54, 22], [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                   [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], np.float32) / 64


def expo_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def expo_io(x):
    x = min(max(x, 0.0), 1.0)
    if x in (0.0, 1.0):
        return x
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2


def _affine(img, s, src_c, dst_c):
    """把 img 绕 src_c 缩放 s 倍，再把 src_c 放到 dst_c。"""
    M = np.float32([[s, 0, dst_c[0] - s * src_c[0]], [0, s, dst_c[1] - s * src_c[1]]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def glitch(A, B, k, f, acc, seed=3, amp=1.0):
    """k ∈ [-1, 1]：<0 切点前（底 = A），≥0 切点后（底 = B）；|k|→0 最强。"""
    env = max(0.0, 1 - abs(k)) ** 1.4 * amp
    base, other = (A, B) if k < 0 else (B, A)
    if env <= 0.01:
        return base
    rng = np.random.default_rng(seed * 1000 + f // 2)
    out = base.copy()
    y = 0
    while y < H:                                           # 切片：错位 / 换成另一镜头
        h = int(rng.integers(int(6 * R), int(80 * R)))
        u = rng.random()
        if u < 0.45 * env:
            out[y:y + h] = np.roll(base[y:y + h], int(rng.normal(0, 140 * R * env)), axis=1)
        elif u < 0.70 * env:
            out[y:y + h] = np.roll(other[y:y + h], int(rng.normal(0, 60 * R * env)), axis=1)
        y += h
    bs = int(32 * R)                                       # 宏块位移
    for _ in range(int(60 * env)):
        bx, by = int(rng.integers(0, W // bs)) * bs, int(rng.integers(0, H // bs)) * bs
        w_, h_ = bs * int(rng.integers(1, 5)), bs * int(rng.integers(1, 3))
        sx = int(np.clip(bx + rng.normal(0, 90 * R), 0, W - w_)); sy = int(np.clip(by + rng.normal(0, 20 * R), 0, H - h_))
        if bx + w_ <= W and by + h_ <= H:
            out[by:by + h_, bx:bx + w_] = out[sy:sy + h_, sx:sx + w_]
    px = int(16 * R * env)                                 # RGB 分离
    if px:
        out[..., 0] = np.roll(out[..., 0], px, axis=1)
        out[..., 2] = np.roll(out[..., 2], -px, axis=1)
    for _ in range(int(7 * env)):                          # 强调色细块（≤ 约 2% 画面）
        bw, bh = int(rng.uniform(40, 260) * R), int(rng.uniform(3, 14) * R)
        bx, by = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
        col = acc["main"] if rng.random() < 0.6 else acc["sec"]
        out[by:by + bh, bx:bx + bw] = (out[by:by + bh, bx:bx + bw] * 0.25 + np.array(col) * 0.75).astype(np.uint8)
    if env > 0.4:                                          # 扫描线压暗（每 3 行一条，轻）
        out[::3] = (out[::3] * (1 - 0.25 * env)).astype(np.uint8)
    return out


def slam(img, k, s0=1.22, center=(W / 2, H / 2)):
    """M7 砸入：k 0..1，比例 s0→1 指数缓出。"""
    s = 1 + (s0 - 1) * (1 - expo_out(k))
    if abs(s - 1) < 1e-3:
        return img
    M = np.float32([[s, 0, center[0] * (1 - s)], [0, s, center[1] * (1 - s)]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- M4 HUD 锁定接力
def lock_zoom(A, B, f, n, target, acc_b, lock=6, win_w=0.24, label="HANDOFF // TARGET 02"):
    """f = 转场内第几帧（0..n-1）；target = A 里的锁定点 (x, y)（眼睛 / 脸 / 刀尖）。
    0..lock：角标咬住窗口、B 在窗口里抖点解码；lock..n：从窗口穿过去。"""
    w0 = W * win_w; h0 = w0 * H / W
    c0 = (float(np.clip(target[0], w0 / 2 + 20 * R, W - w0 / 2 - 20 * R)),
          float(np.clip(target[1], h0 / 2 + 20 * R, H - h0 / 2 - 20 * R)))

    def state(ff):
        u = expo_io((ff - lock) / float(n - lock)) if ff >= lock else 0.0
        z = math.exp(math.log(w0 / W) * (1 - u))           # 窗口宽 / 画面宽：对数插值
        return z, (c0[0] + (W / 2 - c0[0]) * u, c0[1] + (H / 2 - c0[1]) * u)

    z, wc = state(f)
    ww, wh = W * z, H * z
    sA = ww / w0
    if sA > 1.001:                                         # A：绕窗口中心放大 + 径向缩放模糊（5 个样本跨半帧）
        zp, wcp = state(f - 0.6)
        sp = W * zp / w0
        acc = np.zeros((H, W, 3), np.float32)
        for j in range(5):
            t = j / 4.0
            acc += _affine(A, sp + (sA - sp) * t, c0, (wcp[0] + (wc[0] - wcp[0]) * t, wcp[1] + (wc[1] - wcp[1]) * t)).astype(np.float32)
        out = acc / 5
        px = int(min(14 * R, abs(math.log(sA) - math.log(max(sp, 1e-3))) * 60 * R))   # 速度峰值处轻微色散
        if px:
            out[..., 0] = np.roll(out[..., 0], px, axis=1); out[..., 2] = np.roll(out[..., 2], -px, axis=1)
    else:
        out = A.astype(np.float32)
    x0, y0 = int(round(wc[0] - ww / 2)), int(round(wc[1] - wh / 2))
    x1, y1 = int(round(wc[0] + ww / 2)), int(round(wc[1] + wh / 2))
    bx0, by0, bx1, by1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if z >= 0.999:
        out = B.astype(np.float32)
    elif bx1 - bx0 > 2 and by1 - by0 > 2:                  # 窗口里的 B：整帧缩到窗口大小；锁定阶段 Bayer 抖点解码
        Bs = cv2.resize(B, (max(2, x1 - x0), max(2, y1 - y0)), interpolation=cv2.INTER_AREA)
        patch = Bs[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0].astype(np.float32)
        dk = cl((f - 1) / 4.0)
        if dk < 1:
            c = max(1, int(3 * R))
            yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
            on = (BAYER8[(yy // c) % 8, (xx // c) % 8] < dk)[..., None]
            patch = np.where(on, patch, out[by0:by1, bx0:bx1] * 0.35)
        out[by0:by1, bx0:bx1] = patch
    out = np.clip(out, 0, 255).astype(np.uint8)
    if z < 0.995:                                          # 窗口细边 + 角标 + 小字（B 的强调色）
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        pad = (1 - bo(cl(f / 4.0), 2.0)) * 60 * R + 10 * R  # 角标从外 60R 咬进来
        d.rectangle([x0, y0, x1, y1], outline=acc_b["hi"] + (230,), width=max(1, int(1.5 * R)))
        Lb = min(48 * R, ww * 0.18); wl = max(2, int(3 * R))
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cx = (x0 if sx < 0 else x1) + sx * pad; cy = (y0 if sy < 0 else y1) + sy * pad
            d.line([cx, cy, cx - sx * Lb, cy], fill=acc_b["main"] + (255,), width=wl)
            d.line([cx, cy, cx, cy - sy * Lb], fill=acc_b["main"] + (255,), width=wl)
        fm = font(SF.F_MONO, 11 * R)
        tag = "%s   %.3f , %.3f   x%.2f" % (label, wc[0] / W, wc[1] / H, 1 / z)
        tw = d.textlength(tag, font=fm)
        d.rectangle([x0 - pad, y0 - pad - 22 * R, x0 - pad + tw + 12 * R, y0 - pad - 4 * R], fill=acc_b["main"] + (255,))
        d.text((x0 - pad + 6 * R, y0 - pad - 20 * R), tag, font=fm, fill=acc_b["txt"] + (255,))
        b = Image.fromarray(out).convert("RGBA"); b.alpha_composite(lay)
        out = np.asarray(b.convert("RGB"))
    return out


# ---------------------------------------------------------------- M9 档案拍贴
PANELS = [  # (x, y, w, h)，单位 = 画面比例；不规则档案版式，覆盖约 70%
    (0.035, 0.07, 0.30, 0.40), (0.36, 0.05, 0.27, 0.30), (0.655, 0.10, 0.31, 0.36),
    (0.05, 0.52, 0.25, 0.41), (0.33, 0.40, 0.34, 0.50), (0.70, 0.50, 0.26, 0.42),
]


def panel_slap(A, B, f, n, acc_b, titles=None, stagger=2, pop=5, merge_from=None):
    """f = 转场内第几帧（0..n-1）。弹出阶段 0..merge_from，合并阶段 merge_from..n（各格缓动到满屏）。"""
    merge_from = merge_from if merge_from is not None else n // 2
    titles = titles or ["FILE %02d" % (i + 1) for i in range(len(PANELS))]
    out = A.copy()
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    m = eo((f - merge_from) / float(n - merge_from)) if f >= merge_from else 0.0
    ft = font(SF.F_WIDE, 10 * R)
    order = [4, 0, 2, 1, 5, 3]                              # 先中间大格，再四周
    for rank, i in enumerate(order):
        k = (f - rank * stagger) / float(pop)
        if k <= 0:
            continue
        px, py, pw, ph = PANELS[i]
        x0, y0, x1, y1 = px * W, py * H, (px + pw) * W, (py + ph) * H
        s = 0.6 + 0.4 * bo(min(k, 1.0), 2.0)                # 0.6→≈1.06→1
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) / 2 * s, (y1 - y0) / 2 * s
        X0, Y0, X1, Y1 = cx - hw, cy - hh, cx + hw, cy + hh
        X0, Y0, X1, Y1 = X0 * (1 - m), Y0 * (1 - m), X1 + (W - X1) * m, Y1 + (H - Y1) * m     # 合并：扩到满屏
        a0, b0, a1, b1 = int(max(0, X0)), int(max(0, Y0)), int(min(W, X1)), int(min(H, Y1))
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        par = int((1 - eo(min(k, 1.0))) * 20 * R * (1 - m))  # 内容视差回位
        sy0, sy1 = int(np.clip(b0 + par, 0, H)), int(np.clip(b1 + par, 0, H))
        patch = B[sy0:sy1, a0:a1]
        out[b0:b0 + patch.shape[0], a0:a0 + patch.shape[1]] = patch
        if m < 0.98:
            al = int(255 * (1 - m))
            d.rectangle([a0, b0, a1, b1], outline=INK + (al,), width=max(1, int(1.5 * R)))
            if rank < 4:                                    # 标题条（宽度跟着弹出拉开）
                bw = (a1 - a0) * 0.62 * eo(min(1.0, (k * pop - 1) / 4.0))
                if bw > 4:
                    d.rectangle([a0, b0 - 16 * R, a0 + bw, b0], fill=acc_b["main"] + (al,))
                    if bw > 120 * R:
                        d.text((a0 + 5 * R, b0 - 14 * R), titles[i % len(titles)], font=ft, fill=acc_b["txt"] + (al,))
    b = Image.fromarray(out).convert("RGBA"); b.alpha_composite(lay)
    return np.asarray(b.convert("RGB"))


# ---------------------------------------------------------------- M2 甩镜（模糊峰值处切）
def _dirblur(img, px, horizontal=True):
    n = int(px)
    if n < 2:
        return img
    k = np.zeros((1, n) if horizontal else (n, 1), np.float32); k[:] = 1.0 / n
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def whip(A, B, f, n, direction=1):
    """A、B 并成一条横幅，镜头从 A 甩到 B（ease-in-out），方向模糊跟速度走，接缝处最糊（切点就在模糊峰值）。无镜像边。"""
    def pos(ff):
        return expo_io(cl(ff / float(n - 1)))
    x = pos(f) * W
    v = abs(pos(f + 0.5) - pos(f - 0.5)) * W
    strip = np.hstack([A, B]) if direction > 0 else np.hstack([B, A])
    x0 = int(round(x if direction > 0 else W - x))
    out = strip[:, x0:x0 + W]
    return _dirblur(out, min(60 * R, v * 0.9))


# ---------------------------------------------------------------- M5 网格条错位
def grid(A, B, f, n, cols=12, stagger=0.55, acc=None):
    """竖条按右→左的波依次滑走（奇偶列上下交替），露出 B；条边一根强调色细线。无过冲。"""
    out = B.copy()
    cw = int(math.ceil(W / float(cols)))
    for c in range(cols):
        x0, x1 = c * cw, min(W, (c + 1) * cw)
        order = cols - 1 - c
        t0 = order / float(cols) * stagger
        k = cl((f / float(n - 1) - t0) / (1 - stagger))
        e = 0.5 - 0.5 * math.cos(math.pi * k)              # ease-in-out
        dy = int(e * H) * (1 if c % 2 == 0 else -1)
        if abs(dy) >= H:
            continue
        strip = A[:, x0:x1]
        if dy >= 0:
            out[dy:, x0:x1] = strip[:H - dy]
            if acc is not None and 0 < dy < H:
                out[max(0, dy - 2):dy + 1, x0:x1] = acc["main"]
        else:
            out[:H + dy, x0:x1] = strip[-dy:]
            if acc is not None:
                out[H + dy - 1:min(H, H + dy + 2), x0:x1] = acc["main"]
    return out

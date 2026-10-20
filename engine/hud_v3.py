# -*- coding: utf-8 -*-
"""
HUD v3（用户 10-19 第 2 / 4 条）：
  "HUD 重合样式太多，而且就像是 SVG 图层在平移（HUD 本身没有 motion，很僵硬），覆盖时间太长"
  "HUD 追踪和素材结合的有些突兀，就像是'不在一个图层'"
改法：
  · 少：每镜只有 取景框 + 一个后景元素（EVA 蜂巢阵约 45%，其余从元素库抽一个）+ 至多 1 个采样窗；气泡簇 / 角落信息块只偶尔出现
  · 有生命周期：入场（取景框从画外带过冲飞来锁定、后景元素从锚点径向揭开）→ 持续约 0.6 个镜头（≥0.9 s）→ 退场（收缩 + 切片 glitch 消失）
  · 取景框用阻尼弹簧追踪（ζ 0.55 / ω 16），不再是刚性贴死目标
  · 同一层：后景元素跟着镜头运动（光流估的全局相似变换 × 景深系数）、被前景人物遮挡（抠像）、带景深轻模糊；
    整层按画面局部亮度调光、加泛光（加色 screen）、镜头快速运动时带方向模糊；之后和画面一起吃颗粒
"""
import math
import numpy as np
import cv2
from PIL import Image

from style_frames import W, H, R
from cards import eo, bo, cl
import hud_elems as HE
import v3_hex as HX
from hud_v2 import HudKit, Canvas, WHITE, F_WIDE


def premult(cv):
    a = cv.a.astype(np.float32) / 255.0
    t = np.asarray(cv.txt, np.float32) / 255.0
    ta = t[..., 3:4]
    rgb = t[..., :3] * ta + a[..., :3] * (1 - ta)
    al = ta + a[..., 3:4] * (1 - ta)
    return np.concatenate([rgb, al], 2)


def spring_track(centers, zeta=0.55, omega=16.0, fps=60.0, start_off=None):
    """阻尼弹簧追踪：返回每帧弹簧位置（略滞后、带一点过冲）。start_off = 入场时相对目标的起始偏移（锁定飞入）。"""
    dt = 1.0 / fps
    c = np.asarray(centers, np.float64)
    x = c[0] + (np.asarray(start_off) if start_off is not None else 0.0)
    v = np.zeros(2)
    out = []
    for tgt in c:
        for _ in range(2):                                   # 半步积分更稳
            a = omega * omega * (tgt - x) - 2 * zeta * omega * v
            v = v + a * dt / 2; x = x + v * dt / 2
        out.append(x.copy())
    return np.array(out)


def glitch_rows(P, g, seed):
    if g <= 0.02:
        return P
    rng = np.random.default_rng(seed)
    P = P.copy(); y = 0
    while y < H:
        h = int(rng.integers(int(4 * R), int(50 * R)))
        u = rng.random()
        if u < 0.5 * g:
            P[y:y + h] = np.roll(P[y:y + h], int(rng.normal(0, 80 * R * g)), axis=1)
        elif u < 0.6 * g:
            P[y:y + h] = 0
        y += h
    px = int(10 * R * g)
    if px:
        P[..., 0] = np.roll(P[..., 0], px, axis=1); P[..., 2] = np.roll(P[..., 2], -px, axis=1)
    return P


class HudV3(HudKit):
    def __init__(self, seed, title, tags, callouts):
        super().__init__(seed, title, tags, callouts)
        rng = np.random.default_rng(seed * 7 + 101)
        self.back_kind = "hexfield" if rng.random() < 0.45 else HE.pick(rng, 1)[0]
        self.use_cluster = rng.random() < 0.12
        self.use_plus = False
        self.tagline = rng.random() < 0.35
        self.depth = float(rng.uniform(0.65, 0.95))

    def render3(self, base, f, t, bk, bf, box, acc, rms, src=None, life_in=1.0, life_out=0.0, cam=None, cam_vel=(0.0, 0.0),
                fg=None, n_samples=1, burst=False, hud_glitch=0.0, gseed=0, kind="face"):
        if life_out >= 1.0:
            return base
        back, front = Canvas(), Canvas()
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        avoid = (x0 + bw * 0.15, y0 + bh * 0.15, x1 - bw * 0.15, y1 - bh * 0.15)
        if bw > 0.40 * W or bh > 0.60 * H:                   # 特写：锁眼睛一带
            s_ = 0.17 * W
            cx_ = (x0 + x1) / 2; cy_ = y0 + (0.40 if kind != "person" else 0.13) * bh
            box = (cx_ - s_ / 2, cy_ - s_ * 0.40, cx_ + s_ / 2, cy_ + s_ * 0.40)
            x0, y0, x1, y1 = box
        self.burst = burst
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = 1 if cx < W / 2 else -1
        ax = float(np.clip(cx + side * ((x1 - x0) / 2 + 360 * R), 330 * R, W - 330 * R))
        ay = float(np.clip(cy - 60 * R, 260 * R, H - 300 * R))
        # ---- 后景（世界锁定、被前景遮挡）
        if self.back_kind == "hexfield":
            HX.hex_field(back, (ax - 140 * R, ay - 90 * R), f, bk, bf, acc, self.seed, cols=6, rows=4, r=50 * R)
        else:
            HE.draw(back, self.back_kind, (ax, ay), f, bk, bf, acc, self.seed * 13, rms=rms, i0=int(t * 60))
        if self.use_cluster:
            self.cluster(back, (ax, ay), avoid, f, bk, bf, acc, 0.0)
        # ---- 前景：取景框 + 小标签
        self.finder(front, box, f, acc)
        if self.tagline and f >= 6:
            front.text((60 * R, 56 * R), self.title, 20 * R, WHITE, 0.95, fnt=F_WIDE)
            front.rect((60 * R, 84 * R), (60 * R + 90 * R * eo((f - 6) / 8.0), 87 * R), acc["main"], 1.0)
        out = base
        if n_samples and life_out <= 0.0 and src is not None:
            out = self.samples(front, base, src, f, bk, bf, box, avoid, (ax, ay), acc, n_samples)
        import gpu
        if gpu.use():
            M = np.eye(3, dtype=np.float64)
            if life_out > 0:
                sx = 1.0 + 0.10 * life_out
                M = np.array([[sx, 0, ax * (1 - sx)], [0, sx, ay * (1 - sx)], [0, 0, 1]]) @ M
            if cam is not None:
                C = np.eye(3); C[:2] = np.asarray(cam, np.float64)
                Ce = np.eye(3) + self.depth * (C - np.eye(3))
                M = Ce @ M
            rad = eo(life_in) * 1.25 * math.hypot(W, H) * 0.5 if life_in < 1.0 else 1e6
            g = max(hud_glitch, 0.8 * life_out)
            rows = np.zeros(H, np.float32)
            if g > 0.02:
                rng = np.random.default_rng(gseed)
                y = 0
                while y < H:
                    h = int(rng.integers(int(4 * R), int(50 * R))); u = rng.random()
                    if u < 0.5 * g:
                        rows[y:y + h] = rng.normal(0, 80 * R * g)
                    elif u < 0.6 * g:
                        rows[y:y + h] = 1e5
                    y += h
            fgu = None if fg is None else (np.clip(fg, 0, 1) * 255).astype(np.uint8)
            return gpu.hud3(out, back.a, np.asarray(back.txt), front.a, np.asarray(front.txt), np.linalg.inv(M), (ax, ay), rad, 60 * R,
                            1.0 - life_out, 1.0 - life_out, fg=fgu, rows=rows, vel=cam_vel, rgbsplit=10 * R * g if g > 0.02 else 0.0)
        Pb, Pf = premult(back), premult(front)
        # 后景：径向揭开 → 世界锁定 → 遮挡 → 景深模糊
        if life_in < 1.0:
            yy, xx = np.ogrid[0:H, 0:W]
            rr = np.sqrt((xx - ax) ** 2 + (yy - ay) ** 2)
            rad = eo(life_in) * 1.25 * math.hypot(W, H) * 0.5
            Pb *= np.clip((rad - rr) / (60 * R), 0, 1)[..., None]
        if life_out > 0:
            s = 1.0 + 0.10 * life_out
            M = np.float32([[s, 0, ax * (1 - s)], [0, s, ay * (1 - s)]])
            Pb = cv2.warpAffine(Pb, M, (W, H)) * (1 - life_out)
            Pf = glitch_rows(Pf, 0.8 * life_out, gseed + 3) * (1 - life_out)
        if cam is not None:
            I = np.float32([[1, 0, 0], [0, 1, 0]])
            Mc = (I + self.depth * (np.float32(cam) - I)).astype(np.float32)
            Pb = cv2.warpAffine(Pb, Mc, (W, H))
        if fg is not None:
            Pb *= (1 - fg)[..., None]
        Pb = cv2.GaussianBlur(Pb, (0, 0), 0.9 * R)
        P = Pf + Pb * (1 - Pf[..., 3:4])
        if hud_glitch > 0.02:
            P = glitch_rows(P, hud_glitch, gseed)
        vx, vy = cam_vel
        sp = math.hypot(vx, vy)
        if sp > 4:                                           # 镜头快速运动：方向模糊
            n = 5; acc_ = np.zeros_like(P)
            for k in range(n):
                u = (k / (n - 1) - 0.5) * min(sp * 0.8, 40 * R) / max(sp, 1e-3)
                acc_ += cv2.warpAffine(P, np.float32([[1, 0, vx * u], [0, 1, vy * u]]), (W, H))
            P = acc_ / n
        # 整层：按画面局部亮度调光 + 泛光（screen）
        lum = cv2.resize(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), (W // 16, H // 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        lum = cv2.resize(cv2.GaussianBlur(lum, (0, 0), 2), (W, H), interpolation=cv2.INTER_LINEAR)
        Lf = np.clip(0.80 + 0.55 * lum, 0.8, 1.3)[..., None]
        rgb = P[..., :3] * Lf
        o = out.astype(np.float32) / 255 * (1 - P[..., 3:4]) + rgb
        small = cv2.resize(rgb, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.2 * R), (W, H), interpolation=cv2.INTER_LINEAR) * 0.55
        o = 1 - (1 - np.clip(o, 0, 1)) * (1 - np.clip(glow, 0, 1))
        return (np.clip(o, 0, 1) * 255).astype(np.uint8)

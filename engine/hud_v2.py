# -*- coding: utf-8 -*-
"""
v2 战术 HUD / 取景框精细版。用户 10-18："取景框 HUD 需要做的更精致，突出一种机能和繁复美，多参考原帖"。

照 Raf0x00 原帖（https://x.com/Raf0x00/status/2111418735651406324；包内 refs/raf0x00_ref.mp4，
逐帧看过 2.5 / 3.5 / 5.0 / 5.5 / 11.5 / 12.5 / 13.0 s）的设计语汇：
  · 气泡簇：大小不一的细圆互相交叠成簇（不只是同心），夹同心双环；簇边挂 6–10 个小圆组成的"齿轮"小簇
  · 线宽三档：发丝（带断点颗粒）/ 中线 / 粗 C 弧（270°），外加实心圆盘
  · 斜纹圆盘（圆内 45° 条纹，有的半边染点缀色）
  · 光栅条块：一摞 1px 横线，长短不一、两色交替，右缘参差；拍点上横向错位
  · 方块：实心 / 空心、对角阶梯方块串
  · 贯穿画面的斜向长构造线；带引线的小字标注，部分镜像（原片 "FFMPEG->" 是反字）
  · 点缀色面积极小（原片桃色 / 浅蓝）——这里点缀色 = 画面主色强调色（v2_looks.AccentTrack）
  · 运动：元素直接出现 / 消失（不淡入淡出），旋转按拍 15° 步进，数字按 3 帧跳
取景框本体（机能感）：四角粗 L + 内角细 L + 虚线边 + 外侧刻度尺（8R 小格 / 40R 大格带坐标）、左右测距弧带刻度、
  四个向心三角、锁定前角标白 / 强调色 2 帧交替；重元素（粗弧 / 斜纹盘 / 光栅条 / 方块）避开目标框，脸上只过发丝构造线。
画法：OpenCV LINE_AA + shift=4 亚像素，画在全零 RGBA 上 = 预乘 alpha；文字单独一层（PIL）。
"""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw

import style_frames as SF
from style_frames import W, H, R, font
from cards import eo, bo, cl
import hud_elems as HE

SH = 4
SC = 1 << SH
WHITE = (236, 242, 244)
GREY = (150, 158, 164)
DGREY = (92, 98, 104)
F_MONO, F_WIDE = SF.F_MONO, SF.F_WIDE


def _p(x, y):
    return (int(round(x * SC)), int(round(y * SC)))


def _c(rgb, a=1.0):
    return (rgb[0] * a, rgb[1] * a, rgb[2] * a, 255.0 * a)       # 预乘


def _t(px):
    return max(1, int(round(px)))


class Canvas:
    def __init__(self):
        self.a = np.zeros((H, W, 4), np.uint8)
        self.txt = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.txt)

    def circle(self, c, r, col, a=1.0, th=1.0):
        if r > 0.5:
            cv2.circle(self.a, _p(*c), int(round(r * SC)), _c(col, a), _t(th) if th > 0 else -1, cv2.LINE_AA, SH)

    def arc(self, c, r, a0, a1, col, a=1.0, th=1.0):
        if r > 0.5:
            cv2.ellipse(self.a, _p(*c), (int(r * SC), int(r * SC)), 0, a0, a1, _c(col, a), _t(th), cv2.LINE_AA, SH)

    def grain_circle(self, c, r, col, a, th, seed):
        """断点发丝圆：随机长度的弧段 + 小缺口（种子固定，不闪）。"""
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0, 360)
        end = ang + 360
        while ang < end:
            seg = rng.uniform(8, 46); gap = rng.uniform(1.2, 5.0)
            self.arc(c, r, ang, min(end, ang + seg), col, a, th)
            ang += seg + gap

    def line(self, p, q, col, a=1.0, th=1.0):
        cv2.line(self.a, _p(*p), _p(*q), _c(col, a), _t(th), cv2.LINE_AA, SH)

    def dashed(self, p, q, col, a, th, dash, gap):
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        if L < 1:
            return
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
        s = 0.0
        while s < L:
            e = min(L, s + dash)
            self.line((p[0] + ux * s, p[1] + uy * s), (p[0] + ux * e, p[1] + uy * e), col, a, th)
            s = e + gap

    def rect(self, p, q, col, a=1.0, th=-1):
        x0, y0, x1, y1 = int(round(p[0])), int(round(p[1])), int(round(q[0])), int(round(q[1]))
        cv2.rectangle(self.a, (x0, y0), (x1, y1), _c(col, a), th if th < 0 else _t(th))

    def tri(self, pts, col, a=1.0):
        cv2.fillPoly(self.a, [np.array([_p(*p) for p in pts], np.int32)], _c(col, a), cv2.LINE_AA, SH)

    def hatch(self, c, r, col, a=1.0, ang=45.0, pitch=None, half=None):
        """斜纹圆盘：圆内条纹（抗锯齿圆边），half = 只画朝该角度方向的半边。"""
        pitch = pitch or 5 * R
        x0, y0 = int(max(0, c[0] - r - 2)), int(max(0, c[1] - r - 2))
        x1, y1 = int(min(W, c[0] + r + 3)), int(min(H, c[1] + r + 3))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - c[0], yy - c[1]
        cov = np.clip(r - np.hypot(dx, dy) + 0.5, 0, 1)
        t = math.radians(ang)
        u = (dx * math.cos(t) + dy * math.sin(t)) % pitch
        cov *= np.clip(1 - np.abs(u - pitch * 0.25) / (pitch * 0.25), 0, 1) * 1.6
        cov = np.clip(cov, 0, 1)
        if half is not None:
            h = math.radians(half)
            cov *= ((dx * math.cos(h) + dy * math.sin(h)) > 0)
        cov = (cov * a)[..., None]
        col4 = np.array(_c(col, 1.0), np.float32)
        reg = self.a[y0:y1, x0:x1].astype(np.float32)
        self.a[y0:y1, x0:x1] = np.clip(reg * (1 - cov) + col4 * cov, 0, 255).astype(np.uint8)

    def raster(self, x, y, w, h, colA, colB, seed, jit=0.0, a=1.0):
        """光栅条块：1px 横线一摞，长短不一、两色交替，jit = 拍点错位强度。"""
        rng = np.random.default_rng(seed)
        step = max(2, int(round(2 * R)))
        k = 0
        for yy in range(int(y), int(y + h), step):
            ln = w * rng.uniform(0.35, 1.0)
            sx = rng.normal(0, 18 * R) * jit
            col = colA if (k // 2) % 2 == 0 else colB
            x0 = x + sx
            cv2.line(self.a, (int(x0), yy), (int(x0 + ln), yy), _c(col, a), 1)
            k += 1

    def text(self, xy, s, size, col, a=1.0, mirror=False, fnt=None, anchor="la"):
        f = font(fnt or F_MONO, size)
        if not mirror:
            self.d.text(xy, s, font=f, fill=tuple(col) + (int(255 * a),), anchor=anchor)
            return self.d.textlength(s, font=f)
        tw = int(self.d.textlength(s, font=f)) + 4
        im = Image.new("RGBA", (tw, int(size * 1.4)), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, 0), s, font=f, fill=tuple(col) + (int(255 * a),))
        self.txt.alpha_composite(im.transpose(Image.FLIP_LEFT_RIGHT), (int(xy[0]), int(xy[1])))
        return tw

    def glitch(self, g, seed):
        """HUD 层自身的 glitch（炸裂 / 高燃时）：横切片错位、RGB 分离、少量信号丢失条；种子每 2 帧换。"""
        if g <= 0.02:
            return
        rng = np.random.default_rng(seed)
        a = self.a; t = np.array(self.txt)
        y = 0
        while y < H:
            h = int(rng.integers(int(4 * R), int(60 * R)))
            u = rng.random()
            if u < 0.5 * g:
                dx = int(rng.normal(0, 90 * R * g))
                a[y:y + h] = np.roll(a[y:y + h], dx, axis=1); t[y:y + h] = np.roll(t[y:y + h], dx, axis=1)
            elif u < 0.5 * g + 0.1 * g:
                a[y:y + h] = 0; t[y:y + h] = 0
            y += h
        px = int(12 * R * g)
        if px:
            a[..., 0] = np.roll(a[..., 0], px, axis=1); a[..., 2] = np.roll(a[..., 2], -px, axis=1)
        self.txt = Image.fromarray(t, "RGBA"); self.d = ImageDraw.Draw(self.txt)

    def over(self, base):
        import gpu
        if gpu.use():
            return gpu.over(base, self.a, np.asarray(self.txt))
        a = self.a.astype(np.float32)
        out = base.astype(np.float32) * (1 - a[..., 3:4] / 255.0) + a[..., :3]
        t = np.asarray(self.txt, np.float32)
        ta = t[..., 3:4] / 255.0
        out = out * (1 - ta) + t[..., :3] * ta
        return np.clip(out, 0, 255).astype(np.uint8)


def _hit(b, box, pad):
    return not (b[2] < box[0] - pad or b[0] > box[2] + pad or b[3] < box[1] - pad or b[1] > box[3] + pad)


class HudKit:
    def __init__(self, seed, title, tags, callouts):
        self.seed, self.title, self.tags, self.callouts = seed, title, tags, callouts
        rng = np.random.default_rng(seed)
        U = R
        # 气泡簇模板（@1080 px，× R），相对簇锚点
        el = []
        for _ in range(4):
            el.append(dict(k="grain", dx=rng.uniform(-150, 150), dy=rng.uniform(-120, 120), r=rng.uniform(110, 230), th=1.0, a=0.75))
        for _ in range(6):
            el.append(dict(k="ring", dx=rng.uniform(-230, 230), dy=rng.uniform(-180, 180), r=rng.uniform(34, 95), th=1.0, a=0.85))
        el.append(dict(k="double", dx=rng.uniform(-90, 90), dy=rng.uniform(-80, 80), r=74, th=3.0, a=0.9))
        el.append(dict(k="carc", dx=0, dy=0, r=128, th=9.0, span=270, a=1.0, heavy=True))
        el.append(dict(k="marc", dx=rng.uniform(-60, 60), dy=rng.uniform(-60, 60), r=168, th=4.0, span=140, a=0.9, heavy=True))
        for i in range(2):
            el.append(dict(k="hatch", dx=rng.uniform(-200, 200), dy=rng.uniform(-150, 150), r=rng.uniform(18, 32), a=1.0,
                           half=(None if i == 0 else rng.uniform(0, 360)), heavy=True))
        el.append(dict(k="disc", dx=rng.uniform(-200, 200), dy=rng.uniform(-150, 150), r=24, a=1.0, heavy=True))
        a0 = rng.uniform(0, 2 * math.pi)
        el.append(dict(k="gear", dx=290 * math.cos(a0), dy=230 * math.sin(a0), r=42, a=0.9))
        el.append(dict(k="stair", dx=rng.uniform(-260, -150), dy=rng.uniform(100, 200), r=24, a=1.0, heavy=True))
        el.append(dict(k="raster", dx=rng.uniform(-60, 60), dy=-250, w=260, h=26, a=1.0, heavy=True))
        el.append(dict(k="raster", dx=rng.uniform(60, 160), dy=rng.uniform(150, 230), w=170, h=18, a=1.0, heavy=True))
        for _ in range(3):
            el.append(dict(k="sq", dx=rng.uniform(-300, 300), dy=rng.uniform(-240, 240), r=rng.uniform(9, 16), a=0.9,
                           fill=bool(rng.random() < 0.4), heavy=True))
        self.el = el
        self.order = rng.permutation(len(el))
        # 每个镜头抽一套 UI 语汇：取景框 3 种外形、采样窗 3 种外形、气泡簇只在约 40% 镜头出现、另抽 3 种元素库 2 的元素
        FS = ["L", "chamfer", "bracket", "hex", "diamond", "double", "reticle", "ticks"]
        SS = ["L", "chamfer", "bracket", "double", "notch"]
        self.finder_style = FS[int(rng.integers(len(FS)))]
        self.sample_style = SS[int(rng.integers(len(SS)))]
        self.use_cluster = rng.random() < 0.4
        self.extras = HE.pick(rng, 3)
        self.use_plus = rng.random() < 0.5
        self.lines = [rng.uniform(-30, 30) for _ in range(3)]
        self.lines = [self.lines[0] - 12, self.lines[1] + 74, self.lines[2] + 160]

    # ------------------------------------------------------------ 取景框
    def finder(self, cv, box, f, acc, lock_f=14):
        x0, y0, x1, y1 = box
        k = bo(cl(f / float(lock_f)), 1.6)
        pad = 24 * R + (1 - k) * 150 * R
        X0, Y0, X1, Y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        locked = f >= lock_f
        main = WHITE if (locked or (f // 2) % 2 == 0) else acc["main"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        st = self.finder_style
        if st == "L":
            Lc = 58 * R
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):          # 四角粗 L + 内角细 L
                px, py = (X0 if sx < 0 else X1), (Y0 if sy < 0 else Y1)
                cv.line((px, py), (px - sx * Lc, py), main, 1.0, 4 * R)
                cv.line((px, py), (px, py - sy * Lc), main, 1.0, 4 * R)
                ix, iy = px - sx * 12 * R, py - sy * 12 * R
                cv.line((ix, iy), (ix - sx * 22 * R, iy), WHITE, 0.8, 1)
                cv.line((ix, iy), (ix, iy - sy * 22 * R), WHITE, 0.8, 1)
            for (p, q) in (((X0 + Lc + 8 * R, Y0), (X1 - Lc - 8 * R, Y0)), ((X0 + Lc + 8 * R, Y1), (X1 - Lc - 8 * R, Y1)),
                           ((X0, Y0 + Lc + 8 * R), (X0, Y1 - Lc - 8 * R)), ((X1, Y0 + Lc + 8 * R), (X1, Y1 - Lc - 8 * R))):
                cv.dashed(p, q, WHITE, 0.5, 1, 6 * R, 4 * R)
            for (px, py, dx, dy) in ((cx, Y0 - 4 * R, 0, 1), (cx, Y1 + 4 * R, 0, -1), (X0 - 4 * R, cy, 1, 0), (X1 + 4 * R, cy, -1, 0)):
                s_ = 9 * R
                cv.tri([(px, py), (px - dy * s_ - dx * s_ * 1.4, py - dx * s_ - dy * s_ * 1.4), (px + dy * s_ - dx * s_ * 1.4, py + dx * s_ - dy * s_ * 1.4)],
                       acc["main"] if locked else WHITE, 1.0)
        elif st == "chamfer":                                           # 八边形切角框：切角处粗段 + 两侧梯形刻度
            ch = 44 * R
            HE.poly(cv, HE.chamfer_rect(X0, Y0, X1, Y1, ch), WHITE, 0.8, 1.2 * R)
            for (p, q) in (((X0, Y0 + ch), (X0 + ch, Y0)), ((X1 - ch, Y0), (X1, Y0 + ch)), ((X1, Y1 - ch), (X1 - ch, Y1)), ((X0 + ch, Y1), (X0, Y1 - ch))):
                cv.line(p, q, main, 1.0, 5 * R)
            for side in (-1, 1):
                xs = X0 - 14 * R if side < 0 else X1 + 14 * R
                for j in range(9):
                    yy = cy - 64 * R + j * 16 * R
                    cv.line((xs, yy), (xs + side * (14 if j == 4 else 7) * R, yy), WHITE, 0.7, 1)
            HE.diamond(cv, (cx, Y0 - 12 * R), 7 * R, acc["main"] if locked else WHITE, 1.0, fill=locked)
            HE.diamond(cv, (cx, Y1 + 12 * R), 7 * R, acc["main"] if locked else WHITE, 1.0, fill=locked)
        else:                                                           # [ ] 方括号 + 上下菱形 + 中线虚线
            for side in (-1, 1):
                xs = X0 if side < 0 else X1
                cv.line((xs, Y0), (xs, Y1), main, 1.0, 3.5 * R)
                for yy in (Y0, Y1):
                    cv.line((xs, yy), (xs - side * 34 * R, yy), main, 1.0, 3.5 * R)
                cv.line((xs - side * 14 * R, cy), (xs + side * 26 * R, cy), WHITE, 0.8, 1)
            cv.dashed((X0 + 50 * R, Y0), (X1 - 50 * R, Y0), WHITE, 0.45, 1, 3 * R, 6 * R)
            cv.dashed((X0 + 50 * R, Y1), (X1 - 50 * R, Y1), WHITE, 0.45, 1, 3 * R, 6 * R)
            for yy in (Y0 - 14 * R, Y1 + 14 * R):
                HE.diamond(cv, (cx, yy), 8 * R, acc["main"], 1.0, fill=locked)
                HE.diamond(cv, (cx, yy), 13 * R, WHITE, 0.6)
        if st == "hex":                                                 # 六边形框：两条对边加粗、顶点菱形
            r_ = max(X1 - X0, Y1 - Y0) * 0.60
            pts = HE.hexagon((cx, cy), r_, 30)
            HE.poly(cv, pts, WHITE, 0.85, 1.2 * R)
            for k in (0, 3):
                cv.line(pts[k], pts[(k + 1) % 6], main, 1.0, 5 * R)
            for p_ in pts:
                HE.diamond(cv, p_, 6 * R, acc["main"] if locked else WHITE, 1.0, fill=locked)
            HE.poly(cv, HE.hexagon((cx, cy), r_ * 0.86, 30), WHITE, 0.35, 1)
        elif st == "diamond":                                           # 菱形锁：外菱形 + 内虚线菱形 + 顶点刻度
            hw, hh = (X1 - X0) / 2 + 40 * R, (Y1 - Y0) / 2 + 40 * R
            pts = [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]
            HE.poly(cv, pts, WHITE, 0.85, 1.2 * R)
            cv.line(pts[0], pts[1], main, 1.0, 4 * R); cv.line(pts[2], pts[3], main, 1.0, 4 * R)
            inner = [(cx, cy - hh * 0.72), (cx + hw * 0.72, cy), (cx, cy + hh * 0.72), (cx - hw * 0.72, cy)]
            for k in range(4):
                cv.dashed(inner[k], inner[(k + 1) % 4], WHITE, 0.45, 1, 5 * R, 5 * R)
            for p_ in pts:
                cv.line((p_[0] - 10 * R, p_[1]), (p_[0] + 10 * R, p_[1]), WHITE, 0.8, 1)
                cv.line((p_[0], p_[1] - 10 * R), (p_[0], p_[1] + 10 * R), WHITE, 0.8, 1)
        elif st == "double":                                            # 双框错版：白框 + 强调色框错开（套印偏移感）
            cv.rect((X0, Y0), (X1, Y1), WHITE, 0.85, 1)
            ox, oy = 12 * R, 9 * R
            cv.rect((X0 + ox, Y0 + oy), (X1 + ox, Y1 + oy), acc["main"], 0.95, 2)
            for (px, py) in ((X0, Y0), (X1, Y0), (X0, Y1), (X1, Y1)):
                cv.rect((px - 5 * R, py - 5 * R), (px + 5 * R, py + 5 * R), main, 1.0)
            cv.line((X0 - 30 * R, cy), (X0 - 8 * R, cy), WHITE, 0.8, 1); cv.line((X1 + 8 * R, cy), (X1 + 30 * R, cy), WHITE, 0.8, 1)
        elif st == "reticle":                                           # 分段环形准星：4 段弧按拍转、内虚线环、度数刻度
            rr = 0.55 * math.hypot(X1 - X0, Y1 - Y0)
            rot = 15 * (int(f / 9))
            for k in range(4):
                cv.arc((cx, cy), rr, rot + k * 90 + 12, rot + k * 90 + 70, main, 1.0, 3 * R)
            for a0 in range(0, 360, 10):
                cv.arc((cx, cy), rr * 0.82, a0, a0 + 4, WHITE, 0.5, 1)
            for a0 in range(0, 360, 10):
                a_ = math.radians(a0)
                L_ = (12 if a0 % 90 == 0 else 5) * R
                cv.line((cx + (rr + 6 * R) * math.cos(a_), cy + (rr + 6 * R) * math.sin(a_)),
                        (cx + (rr + 6 * R + L_) * math.cos(a_), cy + (rr + 6 * R + L_) * math.sin(a_)), WHITE, 0.6, 1)
            for (dx_, dy_) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv.line((cx + dx_ * rr * 0.35, cy + dy_ * rr * 0.35), (cx + dx_ * rr * 0.75, cy + dy_ * rr * 0.75), WHITE, 0.7, 1)
        elif st == "ticks":                                             # 长十字线 + 刻度 + 四角点
            ext = 110 * R
            for (p_, q_, horiz) in (((X0 - ext, cy), (X1 + ext, cy), True), ((cx, Y0 - ext), (cx, Y1 + ext), False)):
                cv.line(p_, q_, WHITE, 0.55, 1)
                n_ = int((q_[0] - p_[0] if horiz else q_[1] - p_[1]) / (12 * R))
                for j in range(n_ + 1):
                    if horiz:
                        x_ = p_[0] + j * 12 * R
                        if abs(x_ - cx) > 30 * R:
                            cv.line((x_, cy - (7 if j % 5 == 0 else 3) * R), (x_, cy + (7 if j % 5 == 0 else 3) * R), WHITE, 0.6, 1)
                    else:
                        y_ = p_[1] + j * 12 * R
                        if abs(y_ - cy) > 30 * R:
                            cv.line((cx - (7 if j % 5 == 0 else 3) * R, y_), (cx + (7 if j % 5 == 0 else 3) * R, y_), WHITE, 0.6, 1)
            for (px, py) in ((X0, Y0), (X1, Y0), (X0, Y1), (X1, Y1)):
                cv.circle((px, py), 4 * R, main, 1.0, -1)
            cv.circle((cx, cy), 22 * R, acc["main"] if locked else WHITE, 0.9, 1.5 * R)
        if f >= 4:                                                      # 外侧刻度尺（上边）
            stp, big = 8 * R, 40 * R
            n = int((X1 - X0) / stp)
            for i in range(n + 1):
                x = X0 + i * stp; mj = (i * stp) % big < 0.5
                cv.line((x, Y0 - 10 * R - (16 * R if st != "L" else 0)), (x, Y0 - (22 if mj else 15) * R - (16 * R if st != "L" else 0)),
                        WHITE, 0.8 if mj else 0.5, 1)
                if mj and i % 10 == 0:
                    cv.text((x + 2 * R, Y0 - 38 * R - (16 * R if st != "L" else 0)), "%04d" % int(x), 8 * R, GREY)
        if st == "L":                                                   # 只有 L 款带左右测距弧
            rr = 0.5 * math.hypot(X1 - X0, Y1 - Y0) + 26 * R
            sw = 22 * eo((f - 3) / 10.0)
            if sw > 0.5:
                for a_c in (180, 0):
                    cv.arc((cx, cy), rr, a_c - sw, a_c + sw, WHITE, 0.75, 1.5 * R)
                    for dg in np.arange(-sw, sw + 0.01, 2.0):
                        a = math.radians(a_c + dg); L_ = (9 if abs(dg) % 10 < 1 else 5) * R
                        cv.line((cx + rr * math.cos(a), cy + rr * math.sin(a)), (cx + (rr + L_) * math.cos(a), cy + (rr + L_) * math.sin(a)), WHITE, 0.6, 1)
        tag = "TGT-%02d // %s" % (self.seed + 1, self.title)
        tw = cv.d.textlength(tag, font=font(F_MONO, 11 * R))
        ty = Y0 - 62 * R - (16 * R if st != "L" else 0)
        if st == "chamfer":
            HE.fpoly(cv, HE.chamfer_rect(X0, ty, X0 + tw + 24 * R, ty + 20 * R, 10 * R, (0, 1, 0, 1)), acc["main"], 1.0)
        else:
            cv.rect((X0, ty), (X0 + tw + 14 * R, ty + 20 * R), acc["main"], 1.0)
        cv.text((X0 + 7 * R, ty + 2 * R), tag, 11 * R, acc["txt"])
        if locked:
            cv.raster(X1 - 150 * R, Y1 + 12 * R + (14 * R if st != "L" else 0), 150 * R, 14 * R, acc["main"], WHITE, self.seed + 77, 0)
            cv.text((X1 - 150 * R, Y1 + 30 * R + (14 * R if st != "L" else 0)), "LOCK %.3f  %.3f" % (cx / W, cy / H), 9 * R, WHITE)
        return (X0, Y0, X1, Y1)

    # ------------------------------------------------------------ 气泡簇
    def cluster(self, cv, anchor, box, f, bk, bf, acc, jit):
        ax, ay = anchor
        step = 15 * (bk + eo(bf / 6.0))
        for rank, idx in enumerate(self.order):
            if f < (1 if getattr(self, "burst", False) else 2 + rank):   # 逐个直接出现，1 帧一个；爆点时一齐出现
                continue
            e = self.el[idx]
            c = (ax + e["dx"] * R, ay + e["dy"] * R); r = e.get("r", 0) * R
            if e.get("heavy"):
                ext = (e.get("w", 2 * e.get("r", 0)) * R, e.get("h", 2 * e.get("r", 0)) * R)
                bb = (c[0] - ext[0] / 2, c[1] - ext[1] / 2, c[0] + ext[0] / 2, c[1] + ext[1] / 2)
                if _hit(bb, box, 20 * R):
                    continue                                            # 重元素不压目标（脸）
            k = e["k"]
            if k == "grain":
                cv.grain_circle(c, r, WHITE, e["a"] * 0.8, 1, self.seed * 31 + idx)
            elif k == "ring":
                cv.circle(c, r, WHITE, e["a"] * 0.8, 1)
            elif k == "double":
                cv.circle(c, r, WHITE, 0.9, 3 * R); cv.circle(c, r * 0.8, WHITE, 0.7, 1)
            elif k == "carc":
                a0 = 45 + step
                cv.arc(c, r, a0, a0 + e["span"], acc["main"], 1.0, e["th"] * R)
            elif k == "marc":
                a0 = 200 - step * 1.5
                cv.arc(c, r, a0, a0 + e["span"], GREY, 0.95, e["th"] * R)
            elif k == "hatch":
                cv.hatch(c, r, acc["hi"] if e["half"] is not None else WHITE, 1.0, 45, 5 * R, e["half"])
                cv.circle(c, r, WHITE, 0.8, 1)
            elif k == "disc":
                cv.circle(c, r, DGREY, 1.0, -1)
            elif k == "gear":
                rng = np.random.default_rng(self.seed + 5)
                rot = math.radians((f // 5) * 30)
                for j in range(8):
                    a = rot + j * 2 * math.pi / 8 + rng.uniform(-0.3, 0.3)
                    rr = r * rng.uniform(0.3, 0.9)
                    cv.circle((c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)), r * rng.uniform(0.12, 0.34), WHITE, 0.85, 1)
                cv.circle(c, r * 0.18, WHITE, 0.9, 1)
                cv.line((c[0] - r * 0.5, c[1]), (c[0] + r * 0.5, c[1]), WHITE, 0.6, 1)
                cv.line((c[0], c[1] - r * 0.5), (c[0], c[1] + r * 0.5), WHITE, 0.6, 1)
            elif k == "stair":
                for j in range(4):
                    p = (c[0] + j * r, c[1] + j * r)
                    if j == 2:
                        cv.hatch((p[0] + r / 2, p[1] + r / 2), r * 0.7, WHITE, 1.0, 45, 3 * R)
                    else:
                        cv.rect(p, (p[0] + r, p[1] + r), DGREY if j % 2 == 0 else GREY, 1.0)
            elif k == "raster":
                cv.raster(c[0] - e["w"] * R / 2, c[1] - e["h"] * R / 2, e["w"] * R, e["h"] * R, acc["main"], WHITE,
                          self.seed * 7 + idx + (bk if jit > 0 else 0), jit)
            elif k == "sq":
                cv.rect((c[0] - r, c[1] - r), (c[0] + r, c[1] + r), acc["hi"] if e["fill"] else WHITE, 0.9, -1 if e["fill"] else 1)

    def callout(self, cv, anchor, f, acc):
        ax, ay = anchor
        for j, (txt, dx, dy, mirror) in enumerate(self.callouts):
            if f < (1 if getattr(self, "burst", False) else 10 + 3 * j):
                continue
            p = (ax + dx * R * 0.55, ay + dy * R * 0.55)
            q = (ax + dx * R, ay + dy * R)
            cv.circle(p, 3 * R, WHITE, 1.0, -1)
            cv.line(p, q, WHITE, 0.7, 1)
            hx = q[0] + (60 * R if dx >= 0 else -60 * R)
            cv.line(q, (hx, q[1]), WHITE, 0.7, 1)
            s = txt if (getattr(self, "burst", False) or f >= 10 + 3 * j + len(txt) // 3) else txt[:(f - 10 - 3 * j) * 3]
            x = hx + 6 * R if dx >= 0 else hx - 6 * R - cv.d.textlength(txt, font=font(F_WIDE, 11 * R))
            cv.text((x, q[1] - 9 * R), s, 11 * R, WHITE, 1.0, mirror=mirror, fnt=F_WIDE)

    # ------------------------------------------------------------ 画面四周
    def chrome(self, cv, f, t, acc, rms, wav):
        x, y = 60 * R, 56 * R                                           # 左上：标题块
        cv.text((x, y), self.title, 22 * R, WHITE, 1.0, fnt=F_WIDE)
        cv.rect((x, y + 32 * R), (x + 96 * R, y + 35 * R), acc["main"], 1.0)
        cv.raster(x + 104 * R, y + 28 * R, 120 * R, 10 * R, WHITE, acc["main"], self.seed + 3, 0)
        for i, tg in enumerate(self.tags):
            cv.text((x, y + 46 * R + i * 18 * R), tg, 10.5 * R, WHITE, 0.92)
        rng = np.random.default_rng(self.seed + f // 3)
        y2 = y + 46 * R + (len(self.tags) + 0.6) * 18 * R
        for i in range(6):
            cv.text((x, y2 + i * 14 * R), "%02d  %06.2f  %s" % (i, rng.random() * 360, "▮" * int(rng.integers(1, 6))), 8.5 * R, GREY)
        xr = 26 * R                                                     # 左缘竖刻度
        for i, yy in enumerate(np.arange(220 * R, H - 220 * R, 12 * R)):
            mj = i % 5 == 0
            cv.line((xr, yy), (xr + (14 if mj else 7) * R, yy), WHITE, 0.6 if mj else 0.35, 1)
            if mj and i % 10 == 0:
                cv.text((xr + 18 * R, yy - 6 * R), "%03d" % ((i + f // 6) % 1000), 7.5 * R, GREY)
        tx = W - 330 * R                                                # 右上：时间码 + 小齿轮簇
        cv.text((tx, 56 * R), "T+%07.3f" % t, 13 * R, WHITE, 1.0)
        cv.text((tx, 76 * R), "F%05d   %02d/60" % (int(t * 60), int(t * 60) % 60), 9 * R, GREY)
        g = (W - 100 * R, 80 * R)
        rr = np.random.default_rng(self.seed + 9)
        for j in range(7):
            a = math.radians((f // 4) * 20 + j * 51)
            cv.circle((g[0] + 18 * R * math.cos(a), g[1] + 18 * R * math.sin(a)), rr.uniform(4, 11) * R, WHITE, 0.7, 1)
        cv.rect((g[0] - 52 * R, g[1] + 22 * R), (g[0] - 40 * R, g[1] + 34 * R), acc["main"], 1.0)
        cv.rect((g[0] - 36 * R, g[1] + 28 * R), (g[0] - 24 * R, g[1] + 40 * R), WHITE, 0.8, 1)
        for (cx, cy) in ((40 * R, 40 * R), (W - 40 * R, 40 * R), (40 * R, H - 40 * R), (W - 40 * R, H - 40 * R)):
            cv.circle((cx, cy), 8 * R, WHITE, 0.6, 1)                   # 四角套准标
            cv.line((cx - 16 * R, cy), (cx + 16 * R, cy), WHITE, 0.6, 1); cv.line((cx, cy - 16 * R), (cx, cy + 16 * R), WHITE, 0.6, 1)
        x0, yb = 70 * R, H - 76 * R                                     # 左下：RMS 条
        i0 = int(t * 60)
        for k in range(110):
            v = rms[min(len(rms) - 1, max(0, i0 - 110 + k))]
            h = 3 * R + 62 * R * min(1.0, v)
            xx = x0 + k * 3.6 * R
            cv.rect((xx, yb - h), (xx + 1.6 * R, yb), WHITE if k < 106 else acc["main"], 0.75)
        cv.line((x0, yb + 5 * R), (x0 + 110 * 3.6 * R, yb + 5 * R), WHITE, 0.5, 1)
        for k in range(0, 111, 10):
            cv.line((x0 + k * 3.6 * R, yb + 5 * R), (x0 + k * 3.6 * R, yb + 11 * R), WHITE, 0.5, 1)
        cv.text((x0, yb + 14 * R), "RMS %.3f    WIN 110F    T+%07.3f" % (rms[min(i0, len(rms) - 1)], t), 8 * R, GREY)
        pts = np.array([(W * 0.56 + k * (W * 0.38) / len(wav), H - 112 * R - wav[k] * 110 * R) for k in range(len(wav))], np.float32)
        cv2.polylines(cv.a, [np.round(pts * SC).astype(np.int32)], False, _c(acc["hi"], 0.85), _t(1.2 * R), cv2.LINE_AA, SH)
        cv.line((W * 0.56, H - 112 * R), (W * 0.94, H - 112 * R), WHITE, 0.35, 1)

    # ------------------------------------------------------------ 采样取景（用户 10-18："要多用采样取景风格（和 HUD 结合）"）
    def _detail(self, src, avoid, key):
        """画面里细节最多的一块（局部方差最大、不压目标核心），每拍重选一次（同一拍内不跳）。"""
        if getattr(self, "_dkey", None) == key:
            return self._drect
        g = cv2.cvtColor(cv2.resize(src, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
        var = cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2
        sx, sy = W / 64.0, H / 36.0
        best, rect = -1, None
        for yy in range(4, 32, 2):
            for xx in range(5, 59, 2):
                cx, cy = xx * sx, yy * sy
                if avoid[0] - 80 * R < cx < avoid[2] + 80 * R and avoid[1] - 80 * R < cy < avoid[3] + 80 * R:
                    continue
                if var[yy, xx] > best:
                    best, rect = var[yy, xx], (cx, cy)
        self._dkey, self._drect = key, rect
        return rect

    def samples(self, cv, base, src, f, bk, bf, box, avoid, anchor, acc, n=2):
        """1-2 个放大采样窗：目标（脸 / 眼）+ 画面细节区；源处虚线采样框 + 折线引线，窗内实时取样、
        扫描线 / 十字 / 刻度 / 坐标 / 强调色标题条；按拍"重新采样"（2 帧切片错位），入场自上而下扫入 5 帧。"""
        if src is None or f < 4:
            return base
        out = base
        iw, ih = 430 * R, 252 * R
        zones = [(W - 60 * R - iw, 150 * R), (60 * R, 330 * R), (W - 60 * R - iw, H - 330 * R - ih), (60 * R, H - 230 * R - ih)]
        tx0, ty0, tx1, ty1 = box

        def hit(z):
            zx, zy = z
            return not (zx + iw < tx0 - 40 * R or zx > tx1 + 40 * R or zy + ih < ty0 - 40 * R or zy > ty1 + 40 * R)

        def far(z):
            return -math.hypot(z[0] + iw / 2 - anchor[0], z[1] + ih / 2 - anchor[1])
        zs = sorted(zones, key=lambda z: (hit(z), far(z)))[:n]
        x0, y0, x1, y1 = box
        tw_ = min(x1 - x0, W * 0.2); th_ = tw_ * ih / iw
        tcx, tcy = (x0 + x1) / 2, y0 + (y1 - y0) * 0.42
        dc = self._detail(src, avoid, bk)
        rects = [("LOCK", tcx, tcy, tw_, th_)]
        if dc is not None:
            dw = W * 0.13
            rects.append(("DETAIL", dc[0], dc[1], dw, dw * ih / iw))
        fm = font(F_MONO, 9.5 * R)
        for i, ((lab, cx, cy, sw, sh), (zx, zy)) in enumerate(zip(rects, zs)):
            k0 = 4 + 8 * i
            if f < k0:
                continue
            sx0 = float(np.clip(cx - sw / 2, 0, W - sw)); sy0 = float(np.clip(cy - sh / 2, 0, H - sh))
            crop = src[int(sy0):int(sy0 + sh), int(sx0):int(sx0 + sw)]
            if crop.size == 0:
                continue
            ins = cv2.resize(crop, (int(iw), int(ih)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            ins[::3] *= 0.82                                            # 扫描线
            if bf < 2 and f > k0 + 6:                                   # 按拍重新采样：2 帧切片错位
                rng = np.random.default_rng(bk * 7 + i)
                for _ in range(5):
                    ya = int(rng.integers(0, int(ih) - 12)); hh = int(rng.integers(6, 30))
                    ins[ya:ya + hh] = np.roll(ins[ya:ya + hh], int(rng.normal(0, 40 * R)), axis=1)
            reveal = int(ih * cl((f - k0 + 1) / 5.0))                   # 扫入
            zx_, zy_ = int(zx), int(zy)
            if out is base:
                out = base.copy()
            out[zy_:zy_ + reveal, zx_:zx_ + int(iw)] = np.clip(ins[:reveal], 0, 255).astype(np.uint8)
            if reveal < ih:
                cv.line((zx, zy + reveal), (zx + iw, zy + reveal), acc["hi"], 1.0, 2 * R)
            for (p_, q_) in (((sx0, sy0), (sx0 + sw, sy0)), ((sx0 + sw, sy0), (sx0 + sw, sy0 + sh)),
                             ((sx0 + sw, sy0 + sh), (sx0, sy0 + sh)), ((sx0, sy0 + sh), (sx0, sy0))):
                cv.dashed(p_, q_, acc["hi"], 0.9, 1.4 * R, 7 * R, 5 * R)  # 源处虚线采样框
            ex, ey = (zx if zx > sx0 else zx + iw), zy + ih / 2           # 折线引线
            mx = (sx0 + sw if ex > sx0 + sw else sx0)
            cv.line((mx, sy0 + sh / 2), ((mx + ex) / 2, sy0 + sh / 2), WHITE, 0.7, 1)
            cv.line(((mx + ex) / 2, sy0 + sh / 2), ((mx + ex) / 2, ey), WHITE, 0.7, 1)
            cv.line(((mx + ex) / 2, ey), (ex, ey), WHITE, 0.7, 1)
            cv.circle((mx, sy0 + sh / 2), 3 * R, acc["hi"], 1.0, -1)
            sst = self.sample_style                                     # 窗框三种外形（每镜头抽签）
            if sst == "L":
                cv.rect((zx, zy), (zx + iw, zy + ih), WHITE, 0.85, 1)
                Lc = 22 * R
                for (px, py, ddx, ddy) in ((zx, zy, 1, 1), (zx + iw, zy, -1, 1), (zx, zy + ih, 1, -1), (zx + iw, zy + ih, -1, -1)):
                    cv.line((px, py), (px + ddx * Lc, py), acc["main"], 1.0, 3 * R)
                    cv.line((px, py), (px, py + ddy * Lc), acc["main"], 1.0, 3 * R)
            elif sst == "chamfer":
                ch = 26 * R
                HE.poly(cv, HE.chamfer_rect(zx, zy, zx + iw, zy + ih, ch, (1, 0, 1, 0)), WHITE, 0.9, 1.2 * R)
                cv.line((zx, zy + ch), (zx + ch, zy), acc["main"], 1.0, 4 * R)
                cv.line((zx + iw, zy + ih - ch), (zx + iw - ch, zy + ih), acc["main"], 1.0, 4 * R)
                for j in range(6):
                    cv.line((zx + iw + 6 * R, zy + 14 * R + j * 9 * R), (zx + iw + 14 * R, zy + 14 * R + j * 9 * R), WHITE, 0.7, 1)
            elif sst == "bracket":
                for side in (-1, 1):
                    xs = zx if side < 0 else zx + iw
                    cv.line((xs, zy), (xs, zy + ih), acc["main"], 1.0, 3 * R)
                    for yy in (zy, zy + ih):
                        cv.line((xs, yy), (xs - side * 18 * R, yy), acc["main"], 1.0, 3 * R)
                HE.diamond(cv, (zx + iw / 2, zy + ih + 10 * R), 5 * R, WHITE, 0.8)
            elif sst == "double":                                       # 错版双框
                cv.rect((zx, zy), (zx + iw, zy + ih), WHITE, 0.85, 1)
                cv.rect((zx + 9 * R, zy + 7 * R), (zx + iw + 9 * R, zy + ih + 7 * R), acc["main"], 0.9, 2)
            else:                                                       # 缺口框：四边中点留缺口 + 向内小舌
                g_ = 34 * R
                for (p_, q_) in (((zx, zy), (zx + iw / 2 - g_, zy)), ((zx + iw / 2 + g_, zy), (zx + iw, zy)),
                                 ((zx, zy + ih), (zx + iw / 2 - g_, zy + ih)), ((zx + iw / 2 + g_, zy + ih), (zx + iw, zy + ih)),
                                 ((zx, zy), (zx, zy + ih / 2 - g_)), ((zx, zy + ih / 2 + g_), (zx, zy + ih)),
                                 ((zx + iw, zy), (zx + iw, zy + ih / 2 - g_)), ((zx + iw, zy + ih / 2 + g_), (zx + iw, zy + ih))):
                    cv.line(p_, q_, WHITE, 0.9, 1.4 * R)
                for (p_, q_) in (((zx + iw / 2, zy), (zx + iw / 2, zy + 10 * R)), ((zx + iw / 2, zy + ih), (zx + iw / 2, zy + ih - 10 * R)),
                                 ((zx, zy + ih / 2), (zx + 10 * R, zy + ih / 2)), ((zx + iw, zy + ih / 2), (zx + iw - 10 * R, zy + ih / 2))):
                    cv.line(p_, q_, acc["main"], 1.0, 3 * R)
            mag = iw / max(sw, 1)
            ttl = "SAMPLE-%02d // %s  x%.1f" % (i + 1 + (bk % 7) * 2, lab, mag)
            ttw = cv.d.textlength(ttl, font=fm)
            cv.rect((zx, zy - 20 * R), (zx + ttw + 14 * R, zy - 2 * R), acc["main"], 1.0)
            cv.text((zx + 7 * R, zy - 18.5 * R), ttl, 9.5 * R, acc["txt"])
            ccx, ccy = zx + iw / 2, zy + ih / 2                         # 窗心十字
            cv.line((ccx - 16 * R, ccy), (ccx - 5 * R, ccy), WHITE, 0.7, 1); cv.line((ccx + 5 * R, ccy), (ccx + 16 * R, ccy), WHITE, 0.7, 1)
            cv.line((ccx, ccy - 16 * R), (ccx, ccy - 5 * R), WHITE, 0.7, 1); cv.line((ccx, ccy + 5 * R), (ccx, ccy + 16 * R), WHITE, 0.7, 1)
            for j in range(0, 44):                                      # 底边刻度
                xx = zx + 10 * R + j * (iw - 20 * R) / 43
                cv.line((xx, zy + ih), (xx, zy + ih + (8 if j % 5 == 0 else 4) * R), WHITE, 0.6, 1)
            cv.text((zx, zy + ih + 12 * R), "SRC %.3f , %.3f   %dx%d   F%02d" % (cx / W, cy / H, int(sw), int(sh), f % 100), 8.5 * R, GREY)
        return out

    def render(self, base, f, t, bk, bf, box, acc, rms, wav, tear=0.0, kind="face", burst=False, src=None, n_samples=2,
               hud_glitch=0.0, gseed=0):
        """box = 检测 / 跟踪到的目标框；kind = face / head / person / manual。burst = 高潮爆点：簇元素同时出现。
        src = 原画（采样窗取样用）；n_samples = 采样窗个数（0 = 不开）。"""
        cv = Canvas()
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        avoid = (x0 + bw * 0.15, y0 + bh * 0.15, x1 - bw * 0.15, y1 - bh * 0.15)   # 重元素只避开眼口核心
        if bw > 0.40 * W or bh > 0.60 * H:                 # 特写：锁眼睛一带，不框整个画面
            s_ = 0.17 * W
            cx_ = (x0 + x1) / 2; cy_ = y0 + (0.40 if kind != "person" else 0.13) * bh
            box = (cx_ - s_ / 2, cy_ - s_ * 0.40, cx_ + s_ / 2, cy_ + s_ * 0.40)
            x0, y0, x1, y1 = box
        self.burst = burst
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if f >= 1:                                                      # 贯穿构造线（发丝，脸上只过这个）
            for j, ang in enumerate(self.lines):
                if f < 1 + 2 * j:
                    continue
                a = math.radians(ang); L = 3000 * R
                cv.line((cx - L * math.cos(a), cy - L * math.sin(a)), (cx + L * math.cos(a), cy + L * math.sin(a)), WHITE, 0.28, 1)
        side = 1 if cx < W / 2 else -1                                  # 簇放在空的一侧
        bw = x1 - x0
        ax = cx + side * (bw / 2 + 330 * R); ay = cy - 40 * R
        ax = float(np.clip(ax, 320 * R, W - 320 * R)); ay = float(np.clip(ay, 300 * R, H - 300 * R))
        free = [(ax, ay)] + [(W * 0.50, H * 0.15), (W * 0.24, H * 0.50), (W * 0.76, H * 0.50), (W * 0.50, H * 0.84),
                             (W * 0.30, H * 0.22), (W * 0.70, H * 0.78)]
        free = [q for q in free if not (x0 - 260 * R < q[0] < x1 + 260 * R and y0 - 200 * R < q[1] < y1 + 200 * R)] or free[:1]
        if self.use_cluster:
            self.cluster(cv, (ax, ay), avoid, f, bk, bf, acc, tear)
            self.callout(cv, (ax, ay), f, acc)
            free = free[1:] if free and free[0] == (ax, ay) else free
        rng_ = np.random.default_rng(self.seed + 3)
        for j, name in enumerate(self.extras):                          # 元素库 2：逐个直接出现（不淡入）
            if j < len(free) and f >= 3 + 4 * j:
                HE.draw(cv, name, free[j], f, bk, bf, acc, self.seed * 13 + j, rms=rms, i0=int(t * 60))
        if self.use_plus:
            qx = 0 if (x0 + x1) / 2 > W / 2 else W / 2
            HE.plus(cv, (qx + 40 * R, H * 0.55, qx + W / 2 - 40 * R, H - 120 * R), f, acc, self.seed)
        self.finder(cv, box, f, acc)
        self.chrome(cv, f, t, acc, rms, wav)
        if n_samples:
            base = self.samples(cv, base, src, f, bk, bf, box, avoid, (ax, ay), acc, n_samples)
        cv.glitch(hud_glitch, gseed)
        return cv.over(base)

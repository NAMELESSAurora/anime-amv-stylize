# -*- coding: utf-8 -*-
"""
设计卡渲染（v1 成片）：data/edl_v3.json 里带 card 的镜头 → 2560×1440 / 60 fps PNG 序列。

风格帧（style_frames.py，用户认可的 A/B/C/D）只是一张图；这里把它们拆成"底板 + 动画层"，逐帧合成：
  chapter  EVA 式章节卡：黑底，压缩 Bodoni Black 大字盖章入场（切片错位 4 帧回正），红色罗马数字，
           细线左→右画出，日文副标逐字出现，尾 3 帧 RGB 错位切出
  C        战术 HUD：扫描线自上而下把原画面"扫"成青色半调；轨道环步进旋转，锁定框收拢（锁定前红白交替），
           标签逐字打出，左栏数字滚动；底板缓推 4%。live=True 时底板逐帧取活动画面（E01 召唤开场）
  A        档案拼贴：红标题条拉出，引线先画、裁切框弹出（back-out），星座节点逐个连线，红像素块从轮廓外生长
  B        riso 双色：红版、墨版从两侧错位滑入、套准（带一点过冲）；主字从基线升起、红版字晚 2 帧跟上；
           颗粒每 2 帧换一次（印刷"沸腾"）
  D        漫画页：主格砸入（105%→100%）、特写格从右滑入、红墨小格弹出、竖排说明框弹出；集中线一拍二抖动

时间：卡片覆盖合成帧 [floor(t0·60) − 6, ceil(t1·60) + 6)，前后各 6 帧余量给甩切重叠；
动画按"距 t0 的帧数 f"写，f<0 是入场前状态（甩切进来时看到的）。
build_cut.jsx 读 data/cards_manifest.json，把序列放在 startTime = F0/60、不做时间重映射。

输出: media/cards/<id>/<id>_#####.png、data/cards_manifest.json、ae/qa/cards/<id>.jpg（关键帧拼图）
用法: python scripts/cards.py [--only E01_00,E05d_01] [--jobs 7] [--sheet-only]
"""
import argparse, json, math, os, subprocess, sys, time
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
import style_frames as SF
from style_frames import (W, H, R, RED, INK, PAPER, RISO_RED, RISO_INK, RISO_PAPER, font, vfont, squeeze_text,
                          spaced, gray, thresh_bw, edges, lineart, manga_bw)

ROOT = SF.ROOT
FF = SF.FF
FP = os.path.join(os.path.dirname(FF), "ffprobe.exe")
FPS, PAD = 60, 6
OUT = os.path.join(ROOT, "media", "cards")
QA = os.path.join(ROOT, "ae", "qa", "cards")
CYAN = (70, 215, 235)


# ---------------------------------------------------------------- 通用
def cl(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else float(x))


def eo(x):                      # ease-out cubic
    x = cl(x); return 1 - (1 - x) ** 3


def bo(x, s=1.70158):           # back-out（过冲回弹）
    x = cl(x) - 1; return 1 + (s + 1) * x ** 3 + s * x ** 2


def src_path(yid):
    for sub in ("conform_1440", "conform_1440_new"):           # 1440 原帧率版优先（v2 新到的 2K+ 源在 _new）
        d = os.path.join(ROOT, "media", sub)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if yid in f and f.lower().endswith(".mov"):
                return os.path.join(d, f)
    for r in json.load(open(os.path.join(ROOT, "data", "conform_index.json"), encoding="utf-8")):
        if r["yid"] == yid:
            return r.get("path_1440", r["path"])
    raise FileNotFoundError(yid)


def shot_rec(yid, n):
    return {x["n"]: x for x in json.load(open(os.path.join(ROOT, "data", "shots", yid + ".json"), encoding="utf-8"))["shots"]}[n]


def grab(yid, t):
    p = subprocess.run([FF, "-v", "error", "-ss", "%.4f" % t, "-i", src_path(yid), "-frames:v", "1",
                        "-vf", "scale=%d:%d:flags=lanczos" % (W, H), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True)
    return np.frombuffer(p.stdout, np.uint8)[:W * H * 3].reshape(H, W, 3).copy()


class LiveSource:
    """活动底板：按需顺序解码（不把整段读进内存）。get(ts) 返回源时间 ts 处最近的一帧。"""
    def __init__(self, yid, t0, t1):
        path = src_path(yid)
        r = subprocess.run([FP, "-v", "quiet", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                            "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
        a, b = r.split(",")[0].split("/"); self.fps = float(a) / float(b)      # 有的源 ffprobe 输出带尾逗号（"30/1,"）
        self.t0 = max(0.0, t0)
        self.p = subprocess.Popen([FF, "-v", "error", "-threads", "4", "-ss", "%.4f" % self.t0, "-t", "%.4f" % (t1 - self.t0), "-i", path,   # 解码线程封顶：默认 32 线程每个解码器要 ~3 GB 提交内存
                                   "-vf", "scale=%d:%d:flags=lanczos" % (W, H), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                  stdout=subprocess.PIPE)
        self.k, self.cur = -1, None

    def close(self):
        try:
            self.p.stdout.close(); self.p.kill(); self.p.wait(timeout=5)
        except Exception:
            pass

    def get(self, ts):
        want = max(0, int(round((ts - self.t0) * self.fps)))
        while self.k < want:
            buf = self.p.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break                               # 源段读完：保持最后一帧
            self.cur = np.frombuffer(buf, np.uint8).reshape(H, W, 3); self.k += 1
        return self.cur


_HT = {}


def halftone_c(g, pitch, angle=45):
    """style_frames.halftone 的缓存版：网点距离场只和点距 / 角度有关，逐帧复用。"""
    key = (round(pitch, 3), angle)
    if key not in _HT:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        t = math.radians(angle)
        u = (xx * math.cos(t) + yy * math.sin(t)) % pitch - pitch / 2
        v = (-xx * math.sin(t) + yy * math.cos(t)) % pitch - pitch / 2
        _HT[key] = np.sqrt(u * u + v * v) / (pitch * 0.7071)
    dark = 1 - g.astype(np.float32) / 255
    return np.where(_HT[key] < dark, 0, 255).astype(np.uint8)


def push(img, s, c=(W / 2, H / 2)):
    if abs(s - 1) < 1e-4:
        return img
    M = np.float32([[s, 0, c[0] * (1 - s)], [0, s, c[1] * (1 - s)]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def shift(img, dx, dy, border=0):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def glitch(img, g, seed):
    """切出用的切片错位 + RGB 分离。g 0..1。"""
    if g <= 0:
        return img
    rng = np.random.default_rng(seed)
    out = img.copy()
    y = 0
    while y < H:
        h = int(rng.integers(int(18 * R), int(120 * R)))
        if rng.random() < 0.55:
            out[y:y + h] = np.roll(img[y:y + h], int(rng.normal(0, 70 * R * g)), axis=1)
        y += h
    px = int(14 * R * g)
    out[..., 0] = np.roll(out[..., 0], px, axis=1)
    out[..., 2] = np.roll(out[..., 2], -px, axis=1)
    return out


def comp(base, lay):
    b = Image.fromarray(base).convert("RGBA"); b.alpha_composite(lay)
    return np.asarray(b.convert("RGB"))


def typed(text, n):
    return text[:max(0, min(len(text), int(n)))]


def tight(img):
    bb = img.getchannel("A").getbbox()
    return img.crop(bb) if bb else img


# ---------------------------------------------------------------- chapter（EVA 式）
class Chapter:
    def __init__(self, cd, sh):
        self.cd = cd
        size = 300 * R
        while True:
            fb = font(SF.F_BODONI_BLK, size)
            t = tight(squeeze_text(cd["title"], fb, (246, 242, 236), sx=0.70, track=-4 * R))
            n = tight(squeeze_text(cd["num"], fb, RED, sx=0.80, track=-2 * R))
            gap = int(60 * R)
            if n.width + gap + t.width <= 0.80 * W or size < 120:
                break
            size *= 0.94
        hh = max(t.height, n.height)
        blk = Image.new("RGBA", (n.width + gap + t.width, hh), (0, 0, 0, 0))
        blk.alpha_composite(n, (0, hh - n.height)); blk.alpha_composite(t, (n.width + gap, hh - t.height))
        self.blk = blk
        self.x0 = int(0.11 * W)
        self.y0 = int(0.50 * H - hh * 0.62)
        self.fj = vfont(SF.F_SERIF_CJK, 54 * R, "Bold")
        self.fl = font(SF.F_BAHN, 24 * R)
        self.fm = font(SF.F_MONO, 17 * R)

    def frame(self, f, T, idx):
        img = np.full((H, W, 3), 6, np.uint8)
        if f < 0:
            return img
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        k = eo(f / 4.0)
        s = 1.10 - 0.10 * k
        b = self.blk.resize((max(1, int(self.blk.width * s)), max(1, int(self.blk.height * s))), Image.BILINEAR)
        bx = int(self.x0 - (b.width - self.blk.width) / 2); by = int(self.y0 - (b.height - self.blk.height) / 2)
        if k < 1:                                       # 盖章：4 条横切片错位、回正
            n = 4; sh_ = b.height // n
            jit = [1.0, -0.7, 0.45, -1.0]
            for i in range(n):
                strip = b.crop((0, i * sh_, b.width, b.height if i == n - 1 else (i + 1) * sh_))
                lay.alpha_composite(strip, (bx + int(jit[i] * 70 * R * (1 - k)), by + i * sh_))
        else:
            lay.alpha_composite(b, (bx, by))
        x0, y0, wid, hh = self.x0, self.y0, self.blk.width, self.blk.height
        d.text((x0, y0 - 52 * R), typed("CHAPTER", f - 3), font=self.fl, fill=(150, 150, 156, 255))
        ry = y0 + hh + 34 * R
        rw = wid * eo((f - 2) / 12.0)
        if rw > 1:
            d.line([x0, ry, x0 + rw, ry], fill=(236, 232, 226, 255), width=max(2, int(2 * R)))
        sub = self.cd["sub"]
        nch = int((f - 6) / 3.0) + 1 if f >= 6 else 0
        x = x0
        for ch in typed(sub, nch):
            d.text((x, ry + 26 * R), ch, font=self.fj, fill=(240, 236, 230, 255)); x += d.textlength(ch, font=self.fj) + 14 * R
        lab = "PROVANT  //  CHAPTER %s" % self.cd["num"]
        tl = typed(lab, (f - 10) * 2)
        if tl:
            d.text((x0 + wid - d.textlength(lab, font=self.fm), ry + 40 * R), tl, font=self.fm, fill=(140, 140, 146, 255))
        out = comp(img, lay)
        out = push(out, 1.0 + 0.03 * cl(f / max(T, 1)), (x0 + wid / 2, y0 + hh / 2))
        if T - 3 <= f < T:
            out = glitch(out, (f - (T - 3) + 1) / 3.0, idx)
        return out


# ---------------------------------------------------------------- C 战术 HUD
class Tactical:
    def __init__(self, cd, sh):
        self.cd = cd
        self.cx, self.cy = cd["target"][0] * W, cd["target"][1] * H
        self.fw = font(SF.F_WIDE, 15 * R); self.fm = font(SF.F_MONO, 13 * R); self.fb = font(SF.F_WIDE, 26 * R)
        self.fs = font(SF.F_MONO, 11 * R)
        self.left = self.cx > 0.6 * W                   # 目标偏右时字放左边（E01 令咒在右下）
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def base(self, src):
        g = self.clahe.apply(gray(src))
        g = (255 * (g / 255.0) ** 0.6).astype(np.uint8)
        ht = halftone_c(255 - g, 8 * R, 15)
        tint = np.array(CYAN, np.float32)
        b = np.where(ht[..., None] == 0, (tint * (0.35 + 0.65 * g[..., None] / 255)).clip(0, 255),
                     np.array([6, 10, 16], np.float32)).astype(np.uint8)
        b[edges(src, 40, 110, 2) > 0] = (200, 245, 250)
        return b

    def frame(self, f, T, idx, src, base):
        cx, cy = self.cx, self.cy
        s = 1.0 + 0.04 * cl(f / max(T, 1))
        b = push(base, s, (cx, cy))
        o = push((src.astype(np.float32) * 0.85).astype(np.uint8), s, (cx, cy))
        if f < 0:
            return o
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        if f < 8:                                       # 扫描线：上方已转成 HUD，下方还是原画面
            y = int(H * eo(f / 8.0))
            out = o.copy(); out[:y] = b[:y]
            for w_, a in ((int(14 * R), 50), (int(5 * R), 120), (max(2, int(2 * R)), 255)):
                d.rectangle([0, y - w_ // 2, W, y + w_ // 2], fill=(150, 240, 250, a))
        else:
            out = b
        C = (210, 240, 245, 230); Cd = (120, 170, 180, 160)
        if f >= 1:
            rf = 0.55 + 0.45 * bo(f / 10.0)
            for i, (r, w_, a0, a1) in enumerate(((150, 2, 0, 360), (230, 1, 20, 300), (310, 3, 200, 330), (420, 1, 0, 360), (520, 1, 60, 140))):
                rr = r * R * rf
                rot = (int(f) // 4) * 12 * (1 if i % 2 else -1)
                d.arc([cx - rr, cy - rr, cx + rr, cy + rr], a0 + rot, a1 + rot, fill=C if w_ > 1 else Cd, width=max(1, int(w_ * R)))
            for k in range(72):
                a = math.radians(k * 5 - 0.6 * f); r0 = 420 * R * rf; r1 = r0 + (18 if k % 6 == 0 else 8) * R
                d.line([cx + r0 * math.cos(a), cy + r0 * math.sin(a), cx + r1 * math.cos(a), cy + r1 * math.sin(a)], fill=Cd, width=max(1, int(R)))
        if f >= 2:                                      # 准星十字线向外伸
            L = 1700 * R * eo((f - 2) / 8.0); gp = 60 * R
            for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                d.line([cx + sx * gp, cy + sy * gp, cx + sx * (gp + L), cy + sy * (gp + L)], fill=(120, 170, 180, 110), width=max(1, int(R)))
        if f >= 4:                                      # 锁定框收拢
            dd = 110 * R + 220 * R * (1 - eo((f - 4) / 10.0))
            locked = f >= 14
            col = RED + (255,) if (locked or int(f / 2) % 2 == 0) else (245, 250, 250, 255)
            Lb = 60 * R
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                x0, y0 = cx + sx * dd, cy + sy * dd
                d.line([x0, y0, x0 - sx * Lb, y0], fill=col, width=int(3 * R))
                d.line([x0, y0, x0, y0 - sy * Lb], fill=col, width=int(3 * R))
            fx_, fy_ = cx / W, cy / H
            if not locked:
                rr = np.random.default_rng(int(f))
                fx_, fy_ = rr.random(), rr.random()
            d.text((cx - dd, cy + dd + 10 * R), "X %.3f   Y %.3f" % (fx_, fy_), font=self.fs, fill=C)
            if locked:
                d.text((cx + dd - 120 * R, cy + dd + 10 * R), "[ LOCKED ]", font=self.fs, fill=RED + (255,))
        head = self.cd.get("head", "TARGET LOCK")
        tags = self.cd["tags"]
        if self.left:
            wmax = max([d.textlength(head, font=self.fb)] + [d.textlength(t, font=self.fm) for t in tags])
            tx = cx - 140 * R - wmax
        else:
            tx = cx + 130 * R
        ty = min(cy - 150 * R, H - 260 * R)
        if f >= 6:
            hs = typed(head, f - 6 + 1)
            d.text((tx, ty), hs, font=self.fb, fill=RED + (255,))
            if len(hs) == len(head) and int(f / 8) % 2 == 0:
                xe = tx + d.textlength(hs, font=self.fb) + 8 * R
                d.rectangle([xe, ty + 4 * R, xe + 14 * R, ty + 30 * R], fill=RED + (255,))
        for i, t in enumerate(tags):
            st = 10 + 4 * i
            if f >= st:
                d.text((tx, ty + 40 * R + i * 22 * R), typed(t, (f - st + 1) * 2), font=self.fm, fill=C)
        for i in range(18):
            if f >= i * 0.7:
                y = 80 * R + i * 20 * R
                v = (i * 37.7 + (int(f) // 3) * 13.1) % 360
                d.text((60 * R, y), "%02d  %06.2f  %s" % (i, v, "■" * ((i + int(f) // 6) % 5)), font=self.fm, fill=Cd)
        foot = self.cd.get("foot", "PROVANT // SERVANT ARCHIVE")
        if f >= 8:
            d.text((W - 640 * R, H - 90 * R), typed(foot, (f - 8) * 3), font=self.fw, fill=C)
        out = comp(out, lay)
        if T - 3 <= f < T:
            out = glitch(out, (f - (T - 3) + 1) / 3.0, idx)
        return out


# ---------------------------------------------------------------- A 档案拼贴
class Archive:
    def __init__(self, cd, src):
        self.cd = cd
        bw = thresh_bw(src, 0.28, 0.66)
        ht = halftone_c(bw, 7 * R, 45)
        self.base = np.dstack([np.where(ht > 0, PAPER[i], INK[i]) for i in range(3)]).astype(np.uint8)
        # 红像素块：沿主体轮廓外侧，从三个种子点向外长（style_frames 的规则），按距离排序逐帧出现
        e = edges(src, 50, 130, 3)
        cell = int(14 * R); self.cell = cell
        gy, gx = np.mgrid[0:H // cell, 0:W // cell]
        eg = cv2.dilate(e, np.ones((cell, cell), np.uint8))[::cell, ::cell][:gy.shape[0], :gx.shape[1]] > 0
        BAY = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
        best = np.full(gy.shape, 9.0)
        for ox, oy, rad in ((0.18, 0.78, 0.22), (0.80, 0.70, 0.16), (0.30, 0.20, 0.12)):
            dist = np.hypot(gx * cell - W * ox, gy * cell - H * oy) / (H * rad) + BAY[gy % 4, gx % 4] * 0.45
            ok = (dist < 1.0) & (eg | (dist < 0.55))
            best = np.where(ok, np.minimum(best, dist), best)
        ys, xs = np.nonzero(best < 9)
        self.cells = sorted(zip(best[ys, xs], xs * cell, ys * cell))
        self.boxes = []
        for (sx, sy, sw, sh_), (dx, dy, dw, dh), lab in cd["boxes"]:
            crop = Image.fromarray(src[sy:sy + sh_, sx:sx + sw]).convert("L").resize((dw, dh), Image.LANCZOS)
            crop = Image.eval(crop, lambda v: max(0, min(255, int((v - 128) * 1.6 + 118)))).convert("RGBA")
            self.boxes.append(((sx, sy, sw, sh_), (dx, dy, dw, dh), lab, crop))
        self.ft = font(SF.F_WIDE, 13 * R); self.fm = font(SF.F_MONO, 11 * R); self.fb = font(SF.F_WIDE, 30 * R)

    def frame(self, f, T, idx):
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        ft, fm = self.ft, self.fm
        if f >= 4:
            g = eo((f - 4) / 36.0)
            for dist, X, Y in self.cells:
                if dist > g:
                    break
                d.rectangle([X, Y, X + self.cell - 2, Y + self.cell - 2], fill=RED + (240,))
        for j, ((sx, sy, sw, sh_), (dx, dy, dw, dh), lab, crop) in enumerate(self.boxes):
            fj = 8 + 6 * j
            lp = eo((f - fj + 2) / 4.0)
            if lp > 0:
                x0, y0, x1, y1 = sx + sw / 2, sy + sh_ / 2, dx, dy + dh / 2
                d.line([x0, y0, x0 + (x1 - x0) * lp, y0 + (y1 - y0) * lp], fill=RED + (255,), width=max(1, int(R)))
            if f >= fj:
                s = 0.7 + 0.3 * bo((f - fj) / 5.0)
                cw, ch = max(1, int(dw * s)), max(1, int(dh * s))
                ox, oy = int(dx + dw / 2 - cw / 2), int(dy + dh / 2 - ch / 2)
                lay.paste(crop.resize((cw, ch), Image.BILINEAR), (ox, oy))
                d.rectangle([ox, oy, ox + cw, oy + ch], outline=(30, 30, 34, 255), width=max(1, int(R)))
                bar = int(20 * R); bw_ = dw * 1.1 * eo((f - fj - 1) / 4.0)
                if bw_ > 1:
                    d.rectangle([dx, dy - bar, dx + bw_, dy], fill=RED + (255,))
                    if bw_ > dw * 0.8:
                        d.text((dx + 8 * R, dy - bar + 3 * R), lab, font=ft, fill=(255, 245, 240, 255))
        nodes, labels = self.cd["nodes"], self.cd["labels"]
        for i, ((x, y), lb) in enumerate(zip(nodes, labels)):
            fi = 14 + 5 * i
            if i:
                p = eo((f - fi + 2) / 4.0)
                if p > 0:
                    px, py = nodes[i - 1]
                    d.line([px, py, px + (x - px) * p, py + (y - py) * p], fill=(250, 250, 250, 230), width=max(1, int(1.2 * R)))
            if f >= fi:
                r = 5 * R * bo((f - fi) / 3.0)
                d.ellipse([x - r, y - r, x + r, y + r], outline=(250, 250, 250, 255), width=max(1, int(1.4 * R)))
                d.text((x + 10 * R, y - 18 * R), typed(lb, (f - fi + 1) * 2), font=ft, fill=(250, 250, 250, 255))
                d.text((x + 10 * R, y - 2 * R), typed("%.3f , %.3f" % (x / W, y / H), (f - fi - 1) * 3), font=fm, fill=(200, 200, 200, 255))
        bwid = (W * 0.33) * eo(f / 6.0)
        if bwid > 1:
            tb = Image.new("RGBA", (int(W * 0.33) + 1, int(46 * R) + 1), RED + (255,))
            ImageDraw.Draw(tb).text((16 * R, 6 * R), self.cd["title"], font=self.fb, fill=(255, 250, 245, 255))
            lay.alpha_composite(tb.crop((0, 0, int(bwid), tb.height)), (int(60 * R), int(60 * R)))
        out = comp(self.base, lay)
        return push(out, 1.0 + 0.03 * cl(f / max(T, 1)))


# ---------------------------------------------------------------- B riso 双色
class Riso:
    def __init__(self, cd, src):
        self.cd = cd
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray(src)).astype(np.float32) / 255
        g = cv2.GaussianBlur(g, (0, 0), 0.8)
        red_d = np.clip((1 - g) * 0.95, 0, 1)
        ink_d = np.clip((0.42 - g) / 0.42, 0, 1) ** 1.1

        def screen(dens, pitch, ang):
            h_ = (halftone_c((255 * (1 - dens)).astype(np.uint8), pitch, ang) == 0).astype(np.float32)
            return cv2.GaussianBlur(h_, (0, 0), 0.6)

        self.red = screen(red_d, 6 * R, 15)
        self.ink = np.maximum(screen(ink_d, 6 * R, 45), lineart(src))      # 套准偏移在逐帧合成时加
        rg = np.random.default_rng(5)
        self.grain = [rg.normal(0, 6, (H, W)).astype(np.float32) for _ in range(4)]
        self.kr = 1 - np.array(RISO_RED, np.float32) / 255
        self.ki = 0.92 * (1 - np.array(RISO_INK, np.float32) / 255)
        # 排版（style_frames.look_poster 的版式；layout="tr" 把主字移到右上，给左侧人物让位）
        sz = cd.get("size", 1.0)
        fb = font(SF.F_BODONI_BLK, 250 * R * sz)
        self.t_ink = squeeze_text(cd["big"], fb, RISO_INK, sx=0.78, track=-6 * R)
        self.t_red = squeeze_text(cd["big"], fb, RISO_RED, sx=0.78, track=-6 * R)
        self.M = 144
        self.tr = cd.get("layout") == "tr"
        self.base_y = int(0.42 * H) if self.tr else 960
        self.by = int(self.base_y - self.t_ink.height * 0.80)
        self.bx = (W - self.M - self.t_ink.width) if self.tr else self.M
        self.fl = font(SF.F_BAHN, 26 * R); self.fs = font(SF.F_BAHN, 34 * R)
        self.fj = vfont(SF.F_SERIF_CJK, 40 * R, "Bold"); self.fm = font(SF.F_MONO, 15 * R)

    def plates(self, f):
        k = bo(f / 12.0) if f >= 0 else 0.0
        red = shift(self.red, -28 * R * (1 - k), 18 * R * (1 - k))
        ink = shift(self.ink, 3 * R + 40 * R * (1 - k), 2 * R - 26 * R * (1 - k))
        out = np.array(RISO_PAPER, np.float32) * (1 - red[..., None] * self.kr)
        out = out * (1 - ink[..., None] * self.ki)
        out += self.grain[(int(math.floor(f)) // 2) % 4][..., None]
        return out.clip(0, 255).astype(np.uint8)

    def rise(self, img, v):
        """主字从基线升起：只露出基线以上、按 v 上移的部分。"""
        h = img.height
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        layer.alpha_composite(img, (0, int((1 - v) * h)))
        return layer

    def frame(self, f, T, idx):
        out = self.plates(f)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        M = self.M; INKc = RISO_INK + (255,); REDc = RISO_RED + (255,)
        k = bo(f / 12.0) if f >= 0 else 0.0
        if f >= 3:
            vr = eo((f - 5) / 7.0)
            if vr > 0:
                lay.alpha_composite(self.rise(self.t_red, vr), (self.bx + int(5 * R + 24 * R * (1 - k)), self.by + int(4 * R)))
            lay.alpha_composite(self.rise(self.t_ink, eo((f - 3) / 7.0)), (self.bx, self.by))
        tw = self.t_ink.width
        lab = "CLASS  —  No.%s" % self.cd.get("num", "01")
        if f >= 8:
            if self.tr:
                full = spaced(ImageDraw.Draw(Image.new("L", (1, 1))), (0, 0), lab, self.fl, 255, 9 * R)
                spaced(d, (self.bx + tw - full, self.by - 40 * R), typed(lab, (f - 8) * 2), self.fl, INKc, 9 * R)
            else:
                spaced(d, (M + 4, self.by - 40 * R), typed(lab, (f - 8) * 2), self.fl, INKc, 9 * R)
        ry = self.base_y + 26 * R
        rw = tw * eo((f - 9) / 8.0)
        if rw > 1:
            if self.tr:
                d.line([self.bx + tw - rw, ry, self.bx + tw, ry], fill=INKc, width=max(1, int(1.5 * R)))
            else:
                d.line([M, ry, M + rw, ry], fill=INKc, width=max(1, int(1.5 * R)))
        sub = self.cd["sub"].upper()
        if f >= 11:
            if self.tr:
                full = spaced(ImageDraw.Draw(Image.new("L", (1, 1))), (0, 0), sub, self.fs, 255, 10 * R)
                spaced(d, (self.bx + tw - full, ry + 14 * R), typed(sub, (f - 11) * 2), self.fs, INKc, 10 * R)
            else:
                spaced(d, (M + 2, ry + 14 * R), typed(sub, (f - 11) * 2), self.fs, INKc, 10 * R)
        jp = self.cd["jp"]
        if self.tr:
            x0, y0 = W - M - 40 * R, ry + 90 * R
        else:
            x0, y0 = W - M - 40 * R, 220
        shown = 0
        for kk, ch in enumerate(jp):
            fk = 10 + 2 * kk
            if f >= fk:
                drop = 10 * R * (1 - eo((f - fk) / 3.0))
                d.text((x0, y0 + kk * 46 * R - drop), ch, font=self.fj, fill=REDc); shown = kk + 1
        if shown:
            d.line([x0 - 18 * R, y0, x0 - 18 * R, y0 + shown * 46 * R], fill=REDc, width=max(1, int(1.2 * R)))
        L = 34 * R * eo(f / 6.0)
        if L > 1:
            x0_, y0_, x1_, y1_ = M * 0.5, 72, W - M * 0.5, H - 72
            for (x, y, sx, sy) in ((x0_, y0_, -1, -1), (x1_, y0_, 1, -1), (x0_, y1_, -1, 1), (x1_, y1_, 1, 1)):
                d.line([x + sx * 8, y, x + sx * (8 + L), y], fill=INKc, width=max(1, int(1.2 * R)))
                d.line([x, y + sy * 8, x, y + sy * (8 + L)], fill=INKc, width=max(1, int(1.2 * R)))
        if f >= 6:
            d.text((M, 96), "PROVANT // HEROIC SPIRIT FILE", font=self.fm, fill=INKc)
            n = self.cd.get("num", "01")
            d.text((W - M - 360 * R, H - 112), "RISO 2C  /  RED + INK  /  %s - 12" % n, font=self.fm, fill=INKc)
        out = comp(out, lay)
        return push(out, 1.0 + 0.03 * cl(f / max(T, 1)))


# ---------------------------------------------------------------- D 漫画页
class MangaPage:
    def __init__(self, cd, src):
        self.cd = cd
        G, B = int(26 * R), int(7 * R)
        mx, my = int(70 * R), int(60 * R)
        xs, slant = int(W * 0.60), int(150 * R)
        ym = int(H * 0.47)
        xm = lambda y: xs + slant - (2 * slant) * (y - my) / (H - 2 * my)
        self.P = [[(mx, my), (xs + slant, my), (xs - slant, H - my), (mx, H - my)],
                  [(xs + slant + G, my), (W - mx, my), (W - mx, ym), (xm(ym) + G, ym)],
                  [(xm(ym + G) + G, ym + G), (W - mx, ym + G), (W - mx, H - my), (xs - slant + G, H - my)]]
        self.B, self.mx, self.my, self.xs, self.slant = B, mx, my, xs, slant

        def place(crop_rect, poly):
            x, y, w, h = crop_rect
            bx0, by0 = min(p[0] for p in poly), min(p[1] for p in poly)
            bx1, by1 = max(p[0] for p in poly), max(p[1] for p in poly)
            sc = max((bx1 - bx0) / w, (by1 - by0) / h) * 1.04          # 多 4% 给格内滑动 / 弹出
            c = cv2.resize(src[y:y + h, x:x + w], (int(w * sc) + 2, int(h * sc) + 2), interpolation=cv2.INTER_CUBIC)
            canvas = np.full((H, W, 3), 246, np.uint8)
            cy0, cx0 = int((by0 + by1) / 2 - c.shape[0] / 2), int((bx0 + bx1) / 2 - c.shape[1] / 2)
            ya, xa, yb, xb = max(0, cy0), max(0, cx0), min(H, cy0 + c.shape[0]), min(W, cx0 + c.shape[1])
            canvas[ya:yb, xa:xb] = c[ya - cy0:yb - cy0, xa - cx0:xb - cx0]
            return manga_bw(canvas, ref=src)

        main = manga_bw(src)
        self.c1 = []                                    # 集中线 3 个种子，一拍二轮换
        for seed in (3, 11, 23):
            sl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ds = ImageDraw.Draw(sl)
            rng = np.random.default_rng(seed)
            cx, cy = W * 0.34, H * 0.48
            for k in range(240):
                a = rng.uniform(0, 2 * math.pi); r0 = rng.uniform(0.34, 0.52) * H; r1 = 1.4 * W
                w0 = rng.uniform(2, 6) * R
                ds.polygon([(cx + r0 * math.cos(a), cy + r0 * math.sin(a)),
                            (cx + r1 * math.cos(a - w0 / r1 * 40), cy + r1 * math.sin(a - w0 / r1 * 40)),
                            (cx + r1 * math.cos(a + w0 / r1 * 40), cy + r1 * math.sin(a + w0 / r1 * 40))], fill=(250, 250, 250, 140))
            self.c1.append(comp(main, sl))
        self.c2 = place(cd["eye"], self.P[1])
        p3 = place(cd["sub"], self.P[2])
        self.c3 = np.where(p3 > 128, np.array((246, 238, 232), np.uint8), np.array(RED, np.uint8)).astype(np.uint8)
        self.masks = []
        for poly in self.P:
            m = Image.new("L", (W, H), 0); ImageDraw.Draw(m).polygon(poly, fill=255)
            self.masks.append(np.asarray(m))
        cap = cd["caption"]
        fj = vfont(SF.F_SERIF_CJK, 46 * R, "Bold")
        lh, cw = 54 * R, 64 * R
        bw_, bh_ = int(len(cap) * cw + 36 * R), int(max(len(c) for c in cap) * lh + 48 * R)
        box = Image.new("RGBA", (bw_ + 8, bh_ + 8), (0, 0, 0, 0)); db = ImageDraw.Draw(box)
        db.rectangle([2, 2, 2 + bw_, 2 + bh_], fill=(250, 250, 248, 255), outline=(10, 10, 12, 255), width=max(2, int(3 * R)))
        for c_i, col in enumerate(cap):
            x0 = 2 + bw_ - 18 * R - (c_i + 1) * cw + 8 * R
            for k, ch in enumerate(col):
                db.text((x0, 2 + 24 * R + k * lh), ch, font=fj, fill=(10, 10, 12, 255))
        self.cap = box
        self.cap_xy = (int(xs - slant * 0.1 - bw_ - 60 * R), int(my + 56 * R))
        self.f_pg = font(SF.F_BODONI_BLK, 28 * R); self.f_tag = font(SF.F_BAHN, 20 * R)

    def frame(self, f, T, idx):
        page = np.full((H, W, 3), 246, np.uint8)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        ink = (10, 10, 12, 255)
        polys = []
        if f >= 0:                                      # 主格砸入
            s = 1.05 - 0.05 * eo(f / 4.0)
            P = self.P[0]; c = (sum(p[0] for p in P) / 4, sum(p[1] for p in P) / 4)
            M = np.float32([[s, 0, c[0] * (1 - s)], [0, s, c[1] * (1 - s)]])
            img = cv2.warpAffine(self.c1[(int(f) // 2) % 3], M, (W, H), borderMode=cv2.BORDER_REFLECT)
            msk = cv2.warpAffine(self.masks[0], M, (W, H))
            page = np.where(msk[..., None] > 127, img, page)
            polys.append([(c[0] + (x - c[0]) * s, c[1] + (y - c[1]) * s) for x, y in P])
        if f >= 6:                                      # 特写格从右滑入
            dx = 90 * R * (1 - eo((f - 6) / 4.0))
            img = shift(self.c2, dx, 0, 246); msk = shift(self.masks[1], dx, 0, 0)
            page = np.where(msk[..., None] > 127, img, page)
            polys.append([(x + dx, y) for x, y in self.P[1]])
        if f >= 12:                                     # 红墨小格弹出
            s = 0.9 + 0.1 * bo((f - 12) / 4.0)
            P = self.P[2]; c = (sum(p[0] for p in P) / 4, sum(p[1] for p in P) / 4)
            M = np.float32([[s, 0, c[0] * (1 - s)], [0, s, c[1] * (1 - s)]])
            img = cv2.warpAffine(self.c3, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            msk = cv2.warpAffine(self.masks[2], M, (W, H))
            page = np.where(msk[..., None] > 127, img, page)
            polys.append([(c[0] + (x - c[0]) * s, c[1] + (y - c[1]) * s) for x, y in P])
        for poly in polys:
            d.line(poly + [poly[0]], fill=ink, width=self.B, joint="curve")
        if f >= 18:
            s = 0.85 + 0.15 * bo((f - 18) / 4.0)
            cb = self.cap.resize((max(1, int(self.cap.width * s)), max(1, int(self.cap.height * s))), Image.BILINEAR)
            cx, cy = self.cap_xy[0] + self.cap.width / 2, self.cap_xy[1] + self.cap.height / 2
            lay.alpha_composite(cb, (int(cx - cb.width / 2), int(cy - cb.height / 2)))
        if f >= 22:
            d.text((self.mx, H - self.my + 10 * R), "— %s —" % self.cd.get("page", "07"), font=self.f_pg, fill=ink)
            spaced(d, (W - self.mx - 340 * R, H - self.my + 16 * R), self.cd["tag"], self.f_tag, ink, 7 * R)
        out = comp(page, lay)
        return push(out, 1.0 + 0.02 * cl(f / max(T, 1)))


# ---------------------------------------------------------------- 一张卡
def card_frames(sh):
    t0, t1 = sh["t0"], sh["t1"]
    F0 = int(math.floor(t0 * FPS + 1e-6)) - PAD
    F1 = int(math.ceil(t1 * FPS - 1e-6)) + PAD
    return F0, F1


def render(sh, sheet_only=False):
    cd = sh["card"]; cid = cd["id"]
    t0, t1 = sh["t0"], sh["t1"]
    F0, F1 = card_frames(sh)
    T = (t1 - t0) * FPS
    od = os.path.join(OUT, cid); os.makedirs(od, exist_ok=True)
    rec = shot_rec(sh["yid"], sh["n"])
    tsrc = rec["in"] + (rec["out"] - rec["in"]) * cd.get("frac", 0.5)
    typ = cd["type"]
    live = None
    if typ == "chapter":
        obj = Chapter(cd, sh)
    elif typ == "C":
        obj = Tactical(cd, sh)
        if cd.get("live"):
            ts_of = lambda F: sh["src_in"] + min(max(0.0, F / FPS - t0), sh["play"]) * sh["speed"]
            live = LiveSource(sh["yid"], ts_of(F0) - 0.05, ts_of(F1) + 0.1)
        else:
            still = grab(sh["yid"], tsrc); still_base = obj.base(still)
    else:
        src = grab(sh["yid"], tsrc)
        obj = {"A": Archive, "B": Riso, "D": MangaPage}[typ](cd, src)
    keys = {-3, 2, 6, 12, 20, 30, int(T) - 2, int(T) + 2}          # 拼图取这些帧（相对 t0）
    shots = []
    t_start = time.time()
    for i, F in enumerate(range(F0, F1)):
        f = F - t0 * FPS
        if typ == "C":
            if live is not None:
                src = live.get(sh["src_in"] + min(max(0.0, F / FPS - t0), sh["play"]) * sh["speed"])
                img = obj.frame(f, T, i, src, obj.base(src))
            else:
                img = obj.frame(f, T, i, still, still_base)
        else:
            img = obj.frame(f, T, i)
        if not sheet_only:
            cv2.imwrite(os.path.join(od, "%s_%05d.png" % (cid, i)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if int(math.floor(f)) in keys and len(shots) < 8:
            shots.append((int(math.floor(f)), cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)))
    if live is not None:
        live.p.kill()
    # 关键帧拼图
    os.makedirs(QA, exist_ok=True)
    sheet = np.zeros((360 * 2, 640 * 4, 3), np.uint8)
    for k, (fr, im) in enumerate(shots[:8]):
        y, x = (k // 4) * 360, (k % 4) * 640
        sheet[y:y + 360, x:x + 640] = im
        cv2.putText(sheet, "%s f%d" % (cid, fr), (x + 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
    cv2.imwrite(os.path.join(QA, cid + ".jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dict(id=cid, type=typ, dir=od.replace("\\", "/"), first="%s_00000.png" % cid, F0=F0, n=F1 - F0,
                t0=t0, t1=t1, secs=round(time.time() - t_start, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only"); ap.add_argument("--jobs", type=int, default=7); ap.add_argument("--sheet-only", action="store_true")
    a = ap.parse_args()
    E = json.load(open(os.path.join(ROOT, "data", "edl_v3.json"), encoding="utf-8"))
    shots = [sh for sg in E["segments"] for sh in sg["shots"] if sh.get("card")]
    if a.only:
        want = set(a.only.split(","))
        shots = [sh for sh in shots if sh["card"]["id"] in want]
    mp = os.path.join(ROOT, "data", "cards_manifest.json")
    man = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
    with Pool(min(a.jobs, len(shots))) as pool:
        res = pool.starmap(render, [(sh, a.sheet_only) for sh in shots])
    for r in res:
        print("%-8s %-7s F0=%d n=%d  %.1fs" % (r["id"], r["type"], r["F0"], r["n"], r["secs"]))
        if not a.sheet_only:
            man[r["id"]] = r
    if not a.sheet_only:
        json.dump(man, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", OUT, "/", QA)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
风格帧原型：在真实 Fate 帧上做"概念 PV / 高燃混剪"的几种 look，供用户定方向（v2 试片被评"平庸、像基础拼贴"）。

吸收 grokbot 的三份参考学习（research/ref_study，hectoroz / causasui / Raf0x00）+ 我们的漫画网点，
每种 look 都是"画面处理 + 平面设计层（字、线、框、HUD）"一起做，不只是调色。

  A  ARCHIVE   causasui 白底档案拼贴：主画面阈值黑白，裁切特写进框，红标题条 + 引线，像素块生长，星座节点标签
  B  POSTER    risograph 双色印刷：红版 + 墨版细网错位叠印，原画描线，EVA 式压缩 Bodoni 主字 + 版式网格
  C  TACTICAL  Raf0x00 HUD 图纸：暗底，同心轨道 / 刻度 / 准星锁定，主体半调网点，等宽小字坐标
  D  IMPACT    漫画页：三格（斜边主格 + 眼部特写 + 红墨小格），网点只铺阴影，明朝竖排说明框

用法: python scripts/style_frames.py   → ae/qa/style_frames/*.png + 拼图 _sheet.jpg
"""
import json, os, subprocess, math
from io import BytesIO
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = r"D:\A\provant"
FF = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
OUT = os.path.join(ROOT, "ae", "qa", "style_frames")
W, H = 2560, 1440
R = H / 1080.0
FONTS = r"C:\Windows\Fonts"
LF = os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Microsoft\Windows\Fonts")
F_SERIF = os.path.join(FONTS, "BOD_B.TTF")          # Bodoni Bold：职阶 / 真名大字
F_SERIF_I = os.path.join(FONTS, "BOD_BI.TTF")
F_WIDE = os.path.join(LF, "Michroma-Regular.ttf")    # 宽体无衬线：HUD 标签
F_MONO = os.path.join(FONTS, "CascadiaMono.ttf")     # 等宽：坐标 / 参数
F_JP = os.path.join(LF, "SourceHanSansSC-VF.otf")   # 思源黑：日文竖排
F_JP_B = os.path.join(FONTS, "YuGothB.ttc")
F_BODONI_BLK = os.path.join(FONTS, "BOD_BLAR.TTF")      # Bodoni MT Black：EVA 式标题
F_SERIF_CJK = os.path.join(FONTS, "NotoSerifSC-VF.ttf")  # 明朝风（含假名）：竖排说明框
F_BAHN = os.path.join(FONTS, "bahnschrift.ttf")


def vfont(path, size, weight):
    f = ImageFont.truetype(path, int(size))
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def squeeze_text(text, fnt, color, sx=0.80, track=0):
    """EVA 式横向压缩：先按原宽逐字渲染（可调字距），再把宽度缩到 sx。返回 RGBA PIL 图。"""
    dd = ImageDraw.Draw(Image.new("L", (1, 1)))
    w = int(sum(dd.textlength(ch, font=fnt) + track for ch in text)) + 8
    asc, desc = fnt.getmetrics()
    m = Image.new("L", (w, asc + desc + 8), 0); dm = ImageDraw.Draw(m)
    x = 0
    for ch in text:
        dm.text((x, 0), ch, font=fnt, fill=255); x += dm.textlength(ch, font=fnt) + track
    m = m.resize((max(1, int(w * sx)), m.height), Image.LANCZOS)
    img = Image.new("RGBA", m.size, tuple(color) + (0,)); img.putalpha(m)
    return img


def spaced(d, xy, text, fnt, fill, track):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill); x += d.textlength(ch, font=fnt) + track
    return x


def crop_marks(d, x0, y0, x1, y1, L, col, w):
    for (x, y, sx, sy) in ((x0, y0, -1, -1), (x1, y0, 1, -1), (x0, y1, -1, 1), (x1, y1, 1, 1)):
        d.line([x + sx * 8, y, x + sx * (8 + L), y], fill=col, width=w)
        d.line([x, y + sy * 8, x, y + sy * (8 + L)], fill=col, width=w)

RED = (230, 28, 32)
RUST = (196, 78, 40)
GOLD = (236, 196, 92)
INK = (14, 14, 18)
PAPER = (238, 234, 226)

AB = {"EP": "EpLKseoUV6Q", "TC": "tcC9g9dlv58", "DL": "DlyEbc5AMUw", "L0": "L02aPFAhTCk", "PM": "PmOY1wzlRjo",
      "CF": "Cf9uMIffDTQ", "KX": "kxz8k4ycXCA", "7R": "7R9fdZLTYjo", "G9": "G9nHP6aRsjE", "2J": "_2j1V81wuR0"}


def frame(code, frac=0.5):
    y, n = AB[code[:2]], int(code[2:])
    paths = {}
    for r in json.load(open(os.path.join(ROOT, "data", "conform_index.json"), encoding="utf-8")):
        paths.setdefault(r["yid"], r["path"])
    s = {x["n"]: x for x in json.load(open(os.path.join(ROOT, "data", "shots", y + ".json"), encoding="utf-8"))["shots"]}[n]
    t = s["in"] + (s["out"] - s["in"]) * frac
    p = subprocess.run([FF, "-v", "error", "-ss", "%.3f" % t, "-i", paths[y], "-frames:v", "1", "-vf", "scale=%d:%d:flags=lanczos" % (W, H),
                        "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    return np.asarray(Image.open(BytesIO(p.stdout)).convert("RGB"))


def font(path, size, index=0):
    try:
        return ImageFont.truetype(path, int(size), index=index)
    except Exception:
        return ImageFont.truetype(os.path.join(FONTS, "msyh.ttc"), int(size))


def gray(a):
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def halftone(g, pitch, angle=45):
    """AM 网点（和我们 AE 里自制网屏同原理）：点径随暗度。返回 0/255。"""
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = math.radians(angle)
    u = (xx * math.cos(t) + yy * math.sin(t)) % pitch - pitch / 2
    v = (-xx * math.sin(t) + yy * math.cos(t)) % pitch - pitch / 2
    d = np.sqrt(u * u + v * v) / (pitch * 0.7071)
    dark = 1 - g.astype(np.float32) / 255
    return np.where(d < dark, 0, 255).astype(np.uint8)


def thresh_bw(a, lo=0.30, hi=0.62):
    g = gray(a).astype(np.float32) / 255
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    return (np.clip((g - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def edges(a, lo=60, hi=150, thick=2):
    e = cv2.Canny(cv2.GaussianBlur(gray(a), (0, 0), 1.4), lo, hi)
    return cv2.dilate(e, np.ones((thick, thick), np.uint8))


def poster(a, accent=RED, hi_color=GOLD):
    """hectoroz 版画的"细节保留版"：
    v1 三阶硬切 → 脸和头发被吃成大色块；v2 五阶 → 原画脸上的光晕渐变被切成碎色斑（用户："粗糙"）。
    v3：连续渐变映射（黑 → 暗红 → 红 → 橙金 → 奶白）+ 8 阶轻度分阶（色带细，读起来是"印刷"不是"马赛克"），
         meanshift 先平滑掉画面噪点，原画描线用墨色压回，高光边缘烧亮。"""
    sm = cv2.pyrMeanShiftFiltering(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), 8, 18)
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = np.percentile(g, [3, 97])
    t = np.clip((g - lo) / max(1, hi - lo), 0, 1)
    t = np.round(t * 8) / 8.0                                     # 8 阶
    stops = [(0.00, INK), (0.30, tuple(int(c * 0.50) for c in accent)), (0.55, accent), (0.80, (240, 138, 52)), (1.00, (250, 232, 196))]
    out = np.zeros(a.shape, np.float32)
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        m = (t >= t0) & (t <= t1)
        k = ((t - t0) / (t1 - t0))[..., None]
        seg = np.array(c0, np.float32) * (1 - k) + np.array(c1, np.float32) * k
        out = np.where(m[..., None], seg, out)
    e = edges(a, 60, 150, 2)
    out[(e > 0) & (t > 0.85)] = (255, 246, 214)
    L = lineart(a)[..., None]
    out = out * (1 - L) + np.array(INK, np.float32) * L
    return out.clip(0, 255).astype(np.uint8)

def lineart(a, thick=1):
    """原画描线：局部比周围暗很多的细像素（动画的黑色主线）。返回 0..1 线强度。"""
    g = gray(a).astype(np.float32)
    bg = cv2.dilate(g, np.ones((9, 9), np.uint8))               # 局部最亮（1440p 下描线 2–4 px 宽）
    k = np.clip((bg - g - 22) / 40.0, 0, 1)                      # 比周围暗 22 级以上算线
    k *= np.clip((165 - g) / 70.0, 0, 1)                         # 线本身要够暗
    k = np.clip(k * 1.6, 0, 1)
    if thick > 1:
        k = cv2.dilate(k, np.ones((thick, thick), np.uint8))
    return cv2.GaussianBlur(k, (0, 0), 0.6)


def paste_rgba(base, layer):
    """layer: RGBA PIL image same size"""
    b = Image.fromarray(base).convert("RGBA")
    b.alpha_composite(layer)
    return np.asarray(b.convert("RGB"))


# ---------------------------------------------------------------- A ARCHIVE
def look_archive(src, title, boxes, labels, nodes):
    bw = thresh_bw(src, 0.28, 0.66)
    ht = halftone(bw, 7 * R)
    base = np.dstack([np.where(ht > 0, PAPER[i], INK[i]) for i in range(3)]).astype(np.uint8)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    # 像素块生长（锈红），沿主体轮廓外侧，避开中心脸部
    e = edges(src, 50, 130, 3)
    cell = int(14 * R)
    gy, gx = np.mgrid[0:H // cell, 0:W // cell]
    eg = cv2.dilate(e, np.ones((cell, cell), np.uint8))[::cell, ::cell][:gy.shape[0], :gx.shape[1]] > 0
    BAY = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
    on = np.zeros_like(eg)
    for ox, oy, rad in ((0.18, 0.78, 0.22), (0.80, 0.70, 0.16), (0.30, 0.20, 0.12)):
        dist = np.hypot(gx * cell - W * ox, gy * cell - H * oy) / (H * rad) + BAY[gy % 4, gx % 4] * 0.45
        on |= (dist < 1.0) & (eg | (dist < 0.55))
    for y, x in zip(*np.nonzero(on)):
        X, Y = x * cell, y * cell
        d.rectangle([X, Y, X + cell - 2, Y + cell - 2], fill=RED + (240,))
    # 裁切框 + 红标题条 + 引线
    ft = font(F_WIDE, 13 * R); fm = font(F_MONO, 11 * R)
    for (sx, sy, sw, sh), (dx, dy, dw, dh), lab in boxes:
        crop = Image.fromarray(src[sy:sy + sh, sx:sx + sw]).convert("L").resize((dw, dh), Image.LANCZOS)
        crop = Image.eval(crop, lambda v: max(0, min(255, int((v - 128) * 1.6 + 118))))
        lay.paste(crop.convert("RGBA"), (dx, dy))
        d.rectangle([dx, dy, dx + dw, dy + dh], outline=(30, 30, 34, 255), width=max(1, int(R)))
        bar = int(20 * R)
        d.rectangle([dx, dy - bar, dx + int(dw * 1.1), dy], fill=RED + (255,))
        d.text((dx + 8 * R, dy - bar + 3 * R), lab, font=ft, fill=(255, 245, 240, 255))
        d.line([sx + sw / 2, sy + sh / 2, dx, dy + dh / 2], fill=RED + (255,), width=max(1, int(R)))
    # 星座节点 + 标签
    for i, ((x, y), lb) in enumerate(zip(nodes, labels)):
        if i:
            d.line([nodes[i - 1][0], nodes[i - 1][1], x, y], fill=(250, 250, 250, 230), width=max(1, int(1.2 * R)))
        r = 5 * R
        d.ellipse([x - r, y - r, x + r, y + r], outline=(250, 250, 250, 255), width=max(1, int(1.4 * R)))
        d.text((x + 10 * R, y - 18 * R), lb, font=ft, fill=(250, 250, 250, 255))
        d.text((x + 10 * R, y - 2 * R), "%.3f , %.3f" % (x / W, y / H), font=fm, fill=(200, 200, 200, 255))
    # 大标题条
    fb = font(F_WIDE, 30 * R)
    d.rectangle([60 * R, 60 * R, 60 * R + W * 0.33, 60 * R + 46 * R], fill=RED + (255,))
    d.text((76 * R, 66 * R), title, font=fb, fill=(255, 250, 245, 255))
    return paste_rgba(base, lay)


# ---------------------------------------------------------------- B POSTER (risograph)
RISO_RED = (232, 62, 48)
RISO_INK = (24, 30, 52)          # 偏蓝的黑
RISO_PAPER = (242, 236, 224)


def look_poster(src, big, sub, jp, num="01"):
    """v3（用户："色块太大、细节少；字和画面搭配突兀、土"）：不再 posterize，改成 risograph 双色印刷。
      红版 = 全部影调（中间调也有点），细网 6px@1080，15°；
      墨版 = 阴影 + 原画描线，细网 45°，相对红版错位 3px（套色不准的印刷感）；
      纸色底 + 颗粒。细网点保住了脸、头发、铠甲的明暗。
    排版：版式网格 + 层级（一个主字、两级小字、细线、裁切角标）；字也是油墨色、也错位叠印，
    和画面是"同一张印刷品"，不是贴上去的字幕。"""
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray(src)).astype(np.float32) / 255
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    red_d = np.clip((1 - g) * 0.95, 0, 1)
    ink_d = np.clip((0.42 - g) / 0.42, 0, 1) ** 1.1

    def screen(dens, pitch, ang, off=(0, 0)):
        h_ = (halftone((255 * (1 - dens)).astype(np.uint8), pitch, ang) == 0).astype(np.float32)
        if off != (0, 0):
            h_ = np.roll(np.roll(h_, off[1], 0), off[0], 1)
        return cv2.GaussianBlur(h_, (0, 0), 0.6)

    red = screen(red_d, 6 * R, 15)
    ink = screen(ink_d, 6 * R, 45, (int(3 * R), int(2 * R)))
    L = np.roll(np.roll(lineart(src), int(2 * R), 0), int(3 * R), 1)
    ink = np.maximum(ink, L)
    out = np.array(RISO_PAPER, np.float32) * (1 - red[..., None] * (1 - np.array(RISO_RED, np.float32) / 255))
    out = out * (1 - ink[..., None] * 0.92 * (1 - np.array(RISO_INK, np.float32) / 255))
    out = (out + np.random.default_rng(5).normal(0, 6, out.shape[:2]).astype(np.float32)[..., None]).clip(0, 255).astype(np.uint8)

    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    M = 144
    INKc = RISO_INK + (255,); REDc = RISO_RED + (255,)
    fb = font(F_BODONI_BLK, 250 * R)
    t_ink = squeeze_text(big, fb, RISO_INK, sx=0.78, track=-6 * R)
    t_red = squeeze_text(big, fb, RISO_RED, sx=0.78, track=-6 * R)
    by = int(960 - t_ink.height * 0.80)
    lay.alpha_composite(t_red, (M + int(5 * R), by + int(4 * R)))
    lay.alpha_composite(t_ink, (M, by))
    spaced(d, (M + 4, by - 40 * R), "CLASS  —  No.%s" % num, font(F_BAHN, 26 * R), INKc, 9 * R)
    ry = 960 + 26 * R
    d.line([M, ry, M + t_ink.width, ry], fill=INKc, width=max(1, int(1.5 * R)))
    spaced(d, (M + 2, ry + 14 * R), sub.upper(), font(F_BAHN, 34 * R), INKc, 10 * R)
    fj = vfont(F_SERIF_CJK, 40 * R, "Bold")
    x0, y0 = W - M - 40 * R, 220
    for k, ch in enumerate(jp):
        d.text((x0, y0 + k * 46 * R), ch, font=fj, fill=REDc)
    d.line([x0 - 18 * R, y0, x0 - 18 * R, y0 + len(jp) * 46 * R], fill=REDc, width=max(1, int(1.2 * R)))
    crop_marks(d, M * 0.5, 72, W - M * 0.5, H - 72, 34 * R, INKc, max(1, int(1.2 * R)))
    fm = font(F_MONO, 15 * R)
    d.text((M, 96), "PROVANT // HEROIC SPIRIT FILE", font=fm, fill=INKc)
    d.text((W - M - 360 * R, H - 112), "RISO 2C  /  RED + INK  /  %s - 12" % num, font=fm, fill=INKc)
    return paste_rgba(out, lay)


# ---------------------------------------------------------------- C TACTICAL
def look_tactical(src, target, tags):
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray(src))
    g = (255 * (g / 255.0) ** 0.6).astype(np.uint8)            # 抬暗部：暗镜头里的人才看得见
    ht = halftone(255 - g, 8 * R, 15)                           # 亮处出点（青色点 = 光），暗处留黑
    tint = np.array([70, 215, 235], np.float32)
    base = np.where(ht[..., None] == 0, (tint * (0.35 + 0.65 * g[..., None] / 255)).clip(0, 255), np.array([6, 10, 16], np.float32)).astype(np.uint8)
    e = edges(src, 40, 110, 2)
    base[e > 0] = (200, 245, 250)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    cx, cy = target
    C = (210, 240, 245, 230); Cd = (120, 170, 180, 160)
    for r, w_, a0, a1 in ((150, 2, 0, 360), (230, 1, 20, 300), (310, 3, 200, 330), (420, 1, 0, 360), (520, 1, 60, 140)):
        rr = r * R
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], a0, a1, fill=C if w_ > 1 else Cd, width=int(w_ * R))
    for k in range(72):
        a = math.radians(k * 5); r0 = 420 * R; r1 = r0 + (18 if k % 6 == 0 else 8) * R
        d.line([cx + r0 * math.cos(a), cy + r0 * math.sin(a), cx + r1 * math.cos(a), cy + r1 * math.sin(a)], fill=Cd, width=max(1, int(R)))
    L = 60 * R
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        x0, y0 = cx + sx * 110 * R, cy + sy * 110 * R
        d.line([x0, y0, x0 - sx * L, y0], fill=RED + (255,), width=int(3 * R))
        d.line([x0, y0, x0, y0 - sy * L], fill=RED + (255,), width=int(3 * R))
    fw = font(F_WIDE, 15 * R); fm = font(F_MONO, 13 * R); fb = font(F_WIDE, 26 * R)
    d.text((cx + 130 * R, cy - 150 * R), "TARGET LOCK", font=fb, fill=RED + (255,))
    for i, t in enumerate(tags):
        d.text((cx + 130 * R, cy - 110 * R + i * 22 * R), t, font=fm, fill=C)
    for i in range(18):
        y = 80 * R + i * 20 * R
        d.text((60 * R, y), "%02d  %06.2f  %s" % (i, (i * 37.7) % 360, "■" * (i % 5)), font=fm, fill=Cd)
    d.text((W - 640 * R, H - 90 * R), "PROVANT // SERVANT ARCHIVE 01", font=fw, fill=C)
    return paste_rgba(base, lay)


# ---------------------------------------------------------------- D IMPACT (manga page)
def manga_bw(src, ref=None):
    """漫画处理：原画描线为主，网点只铺阴影（v2 的画面处理，保留）。
    ref：取明暗分位数用的参考帧。特写格要传整帧——只按裁切区域算分位数时，
    局部偏暗的眼部特写会被判成"整格都是暗部"，变成一片实心ベタ（v3 第一次出图的右上格）。"""
    sm = cv2.pyrMeanShiftFiltering(cv2.cvtColor(src, cv2.COLOR_RGB2BGR), 10, 22)
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    gref = g if ref is None else cv2.cvtColor(cv2.pyrMeanShiftFiltering(cv2.cvtColor(ref, cv2.COLOR_RGB2BGR), 10, 22), cv2.COLOR_BGR2GRAY)
    q12, q75 = np.percentile(gref, [12, 75])
    q96 = np.percentile(gref, 96)
    tone = 0.18 + 0.82 * np.clip((g.astype(np.float32) - q12) / max(1, q75 - q12), 0, 1)
    ht = halftone((tone * 255).astype(np.uint8), 7 * R)
    bw = np.where(ht > 0, 245, 12).astype(np.float32)
    bw[g < q12] = 12
    bw[g > q96] = 248
    L = lineart(src, thick=2)
    bw = bw * (1 - L) + 12 * L
    return np.dstack([bw.astype(np.uint8)] * 3)


def look_impact(src, caption, tag, eye_rect, sub_rect):
    """v3（用户："D 的文字也土"）：v2 是"整幅图 + 红色斜半边 + 描边大假名"，像模板。
    改成真正的漫画页：三格（主格斜边 + 眼部特写 + 细节格），白色格间、粗黑格线；
    文字按漫画惯例放进竖排说明框（明朝、白底细框），不做描边大字；
    红色只在第三格（角落小格），不再铺半个画面。"""
    page = np.full((H, W, 3), 246, np.uint8)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    G, B = int(26 * R), int(7 * R)
    mx, my = int(70 * R), int(60 * R)
    xs, slant = int(W * 0.60), int(150 * R)
    ym = int(H * 0.47)
    xm = lambda y: xs + slant - (2 * slant) * (y - my) / (H - 2 * my)   # 斜线在 y 处的 x
    P1 = [(mx, my), (xs + slant, my), (xs - slant, H - my), (mx, H - my)]
    P2 = [(xs + slant + G, my), (W - mx, my), (W - mx, ym), (xm(ym) + G, ym)]
    P3 = [(xm(ym + G) + G, ym + G), (W - mx, ym + G), (W - mx, H - my), (xs - slant + G, H - my)]

    def fill_panel(poly, img, base):
        msk = Image.new("L", (W, H), 0); ImageDraw.Draw(msk).polygon(poly, fill=255)
        return np.where(np.asarray(msk)[..., None] > 0, img, base)

    def place(crop_rect, poly):
        x, y, w, h = crop_rect
        bx0, by0 = min(p[0] for p in poly), min(p[1] for p in poly)
        bx1, by1 = max(p[0] for p in poly), max(p[1] for p in poly)
        sc = max((bx1 - bx0) / w, (by1 - by0) / h)
        c = cv2.resize(src[y:y + h, x:x + w], (int(w * sc) + 2, int(h * sc) + 2), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((H, W, 3), 246, np.uint8)
        cy0, cx0 = int((by0 + by1) / 2 - c.shape[0] / 2), int((bx0 + bx1) / 2 - c.shape[1] / 2)
        ya, xa, yb, xb = max(0, cy0), max(0, cx0), min(H, cy0 + c.shape[0]), min(W, cx0 + c.shape[1])
        canvas[ya:yb, xa:xb] = c[ya - cy0:yb - cy0, xa - cx0:xb - cx0]
        return manga_bw(canvas, ref=src)

    main_img = manga_bw(src)
    sl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ds = ImageDraw.Draw(sl)
    rng = np.random.default_rng(3)
    cx, cy = W * 0.34, H * 0.48
    for k in range(240):
        a = rng.uniform(0, 2 * math.pi); r0 = rng.uniform(0.34, 0.52) * H; r1 = 1.4 * W
        w0 = rng.uniform(2, 6) * R
        p1 = (cx + r0 * math.cos(a), cy + r0 * math.sin(a))
        pa = (cx + r1 * math.cos(a - w0 / r1 * 40), cy + r1 * math.sin(a - w0 / r1 * 40))
        pb = (cx + r1 * math.cos(a + w0 / r1 * 40), cy + r1 * math.sin(a + w0 / r1 * 40))
        ds.polygon([p1, pa, pb], fill=(250, 250, 250, 140))
    main_img = paste_rgba(main_img, sl)
    page = fill_panel(P1, main_img, page)
    page = fill_panel(P2, place(eye_rect, P2), page)
    p3 = place(sub_rect, P3)
    red = np.where(p3 > 128, np.array((246, 238, 232), np.uint8), np.array(RED, np.uint8)).astype(np.uint8)
    page = fill_panel(P3, red, page)
    for poly in (P1, P2, P3):
        d.line(poly + [poly[0]], fill=(10, 10, 12, 255), width=B, joint="curve")
    fj = vfont(F_SERIF_CJK, 46 * R, "Bold")
    lh, cw = 54 * R, 64 * R
    bw_, bh_ = len(caption) * cw + 36 * R, max(len(c) for c in caption) * lh + 48 * R
    bx, by = xs - slant * 0.1 - bw_ - 60 * R, my + 56 * R
    d.rectangle([bx, by, bx + bw_, by + bh_], fill=(250, 250, 248, 255), outline=(10, 10, 12, 255), width=max(2, int(3 * R)))
    for c_i, col in enumerate(caption):                              # 右起竖排
        x0 = bx + bw_ - 18 * R - (c_i + 1) * cw + 8 * R
        for k, ch in enumerate(col):
            d.text((x0, by + 24 * R + k * lh), ch, font=fj, fill=(10, 10, 12, 255))
    d.text((mx, H - my + 10 * R), "— 07 —", font=font(F_BODONI_BLK, 28 * R), fill=(10, 10, 12, 255))
    spaced(d, (W - mx - 340 * R, H - my + 16 * R), tag, font(F_BAHN, 20 * R), (10, 10, 12, 255), 7 * R)
    return paste_rgba(page, lay)


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    # A：贞德（FGO Memorial）档案拼贴
    s = frame("L0040")
    jobs.append(("A_ARCHIVE_jeanne", look_archive(
        s, "RULER  //  JEANNE D'ARC",
        boxes=[((1100, 220, 360, 300), (1900, 260, 420, 350), "FACE / RULER"),
               ((900, 700, 420, 360), (160, 760, 420, 360), "BANNER / LUMINOSITE"),
               ((1350, 900, 300, 260), (1960, 820, 360, 300), "ARMOR / SAINT")],
        labels=["CROWN", "BANNER", "BRAID", "GAUNTLET"],
        nodes=[(1250, 240), (1020, 560), (1420, 760), (1180, 1080)])))
    # B：阿尔托莉雅（FGO Memorial）版画海报
    s = frame("L0078")
    jobs.append(("B_POSTER_artoria", look_poster(s, "SABER", "Artoria Pendragon", "約束された勝利の剣", "02")))
    # C：狮心王?（FSF OP，金发持剑青年）战术 HUD
    s = frame("EP125")
    jobs.append(("C_TACTICAL_richard", look_tactical(s, (int(W * 0.52), int(H * 0.40)),
                                                    ["CLASS   SABER", "TRUE NAME  ████████", "PARAM  STR A / AGI B", "NP  ████████████", "STATUS  ACTIVE"])))
    # D：贞德冲锋（FGO 终局）漫画冲击
    s = frame("CF042")
    jobs.append(("D_IMPACT_jeanne", look_impact(s, ["我が神は", "ここにありて"], "LA PUCELLE  /  RULER",
                                                eye_rect=(720, 540, 440, 380), sub_rect=(1740, 860, 760, 440))))
    for name, img in jobs:
        Image.fromarray(img).save(os.path.join(OUT, name + ".png"))
    # 拼图
    sheet = Image.new("RGB", (1280 * 2, 720 * 2))
    f = font(os.path.join(FONTS, "msyh.ttc"), 30)
    for k, (name, img) in enumerate(jobs):
        im = Image.fromarray(img).resize((1280, 720), Image.LANCZOS)
        x, y = (k % 2) * 1280, (k // 2) * 720
        sheet.paste(im, (x, y))
        dd = ImageDraw.Draw(sheet); dd.rectangle([x, y, x + 16 * len(name) + 20, y + 42], fill=(255, 230, 0)); dd.text((x + 8, y + 4), name, font=f, fill=(0, 0, 0))
    sheet.save(os.path.join(OUT, "_sheet.jpg"), quality=88)
    print("->", OUT)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
v4 新风格层（用户 10-19 看过 v3 的第 3 / 4 / 7 条）：
  A  comic_us     美漫：平涂（色相 12 档 / 明度 4 档、饱和拉高）+ 粗墨线 + 大块实黑阴影 + 中间调 Ben-Day 网点 + 纸色；
                  黄色说明框（Bangers 大写）、冲击镜头加拟声字（黄字黑边红投影，斜 8°，弹出）
  G  saint_graph  Fate 灵基卡：背景压暗虚化 + 金色径向光，竖版卡片翻入（金色双边框 / 角饰 / 顶部职阶条 / 底部真名条，
                  名字只用 data/servant_info.json 里核对过的）
  slit            "画幅收窄 + 强调文字"转场（用户："IMPERSONATOR 这个缩窄画幅然后强调文字的转场我很喜欢"）：
                  A 上下黑边快速合拢成一条窄缝（同时轻推近）→ 缝里切到 B、强调字在缝中展开 → 黑边弹开
Bangers：Google Fonts，SIL OFL（assets/fonts/Bangers-Regular.ttf）。
"""
import math, os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import style_frames as SF
from style_frames import W, H, R, font
from cards import eo, bo, cl

ROOT = r"D:\A\provant"
F_BANG = os.path.join(ROOT, "assets", "fonts", "Bangers-Regular.ttf")
INK = (14, 12, 16)
PAPER = np.array((250, 242, 222), np.float32)
YEL = (255, 222, 60)
GOLD = (214, 178, 96)
GOLD_HI = (250, 226, 160)


def expo_in(x):
    x = cl(x)
    return 0.0 if x <= 0 else 2 ** (10 * (x - 1))


def expo_out(x):
    x = cl(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


# ---------------------------------------------------------------- A 美漫（见 v4_comic.py）
from v4_comic import comic_us  # noqa: E402,F401


# ---------------------------------------------------------------- G Fate 灵基卡（v2：每个英灵一套框，用户"card frame 你可以自己设计一套适合英灵的"）
_BG, _FRAME = {}, {}


def _frame(key):
    """assets/gen/frame_<key>.png（无则用通用金框）；返回 (RGBA 图, 中央窗口 bbox)。窗口 = 含画面中心的透明连通域的外接框。"""
    if key in _FRAME:
        return _FRAME[key]
    p = os.path.join(ROOT, "assets", "gen", "frame_%s.png" % key)
    if not os.path.exists(p):
        p = os.path.join(ROOT, "assets", "gen", "card_frame.png")
    if not os.path.exists(p):
        _FRAME[key] = None
        return None
    im = Image.open(p).convert("RGBA")
    a = np.asarray(im)[..., 3]
    hole = (a < 24).astype(np.uint8)
    n, lab = cv2.connectedComponents(hole)
    c = lab[a.shape[0] // 2, a.shape[1] // 2]
    if c:
        ys, xs = np.nonzero(lab == c)
        bb = (xs.min(), ys.min(), xs.max(), ys.max())
    else:
        bb = (int(a.shape[1] * 0.12), int(a.shape[0] * 0.10), int(a.shape[1] * 0.88), int(a.shape[0] * 0.90))
    _FRAME[key] = (im, bb)
    return _FRAME[key]


_FS = {}


MAP_W = {"okita": 0.18}


def _frame_style(size, key, fim, pal):
    """卡框风格化（缓存）：亮度 → 人物配色渐变映射（暗 = 角色暗色、中 = 主色、亮 = 亮色），和原图 72 / 28 混；
    暗部叠 45° 网点、整框叠细扫描线 —— 原素材里的细小瑕疵被统一的印刷质感盖住。"""
    k = (key, size, tuple(pal["main"]))
    if k in _FS:
        return _FS[k]
    fa = np.asarray(fim.resize(size, Image.LANCZOS)).astype(np.float32)
    rgb_, a = fa[..., :3], fa[..., 3:4]
    lum = (rgb_ @ np.array([0.299, 0.587, 0.114], np.float32))[..., None] / 255.0
    dk = np.array(pal["dk"], np.float32) * 0.55; mn = np.array(pal["main"], np.float32); hi_ = np.array(pal["hi"], np.float32)
    mp = np.where(lum < 0.5, dk + (mn - dk) * (lum / 0.5), mn + (hi_ - mn) * ((lum - 0.5) / 0.5))
    wm = MAP_W.get(key, 0.72)                              # 用户点名了母题颜色的框（冲田：蓝羽织 + 樱花 + 武士刀）少映射，保留原色
    out = mp * wm + rgb_ * (1 - wm)
    h_, w_ = out.shape[:2]
    yy, xx = np.mgrid[0:h_, 0:w_].astype(np.float32)
    pitch = 5.0 * R
    u = (xx + yy) / pitch; v = (yy - xx) / pitch
    dots = (np.hypot(u - np.round(u), v - np.round(v)) < 0.33)[..., None] & (lum < 0.42)
    out = np.where(dots, out * 0.62, out)
    out = out * (1 - 0.10 * ((yy.astype(np.int32) // max(2, int(3 * R))) % 2 == 0)[..., None])
    res = Image.fromarray(np.dstack([np.clip(out, 0, 255), a]).astype(np.uint8), "RGBA")
    _FS[k] = res
    return res


def _card_glitch(card, f, seed):
    """卡片 glitch：入场 10 帧大幅切片拼合 + RGB 分离；之后每 36 帧一次 4 帧的小 glitch；常驻 2 px 的 RGB 分离。"""
    a = np.asarray(card).copy()
    h_, w_ = a.shape[:2]
    rng = np.random.default_rng(seed * 131 + (f // 2))
    if f < 10:
        g = 1.0 - f / 10.0
    elif (f % 36) < 4:
        g = 0.35
    else:
        g = 0.0
    if g > 0:
        y = 0
        while y < h_:
            hh = int(rng.integers(int(6 * R), int(60 * R)))
            if rng.random() < 0.55:
                a[y:y + hh] = np.roll(a[y:y + hh], int(rng.normal(0, 90 * R * g)), axis=1)
            y += hh
    sp = int(round((2 + 14 * g) * R))
    a[..., 0] = np.roll(a[..., 0], sp, axis=1); a[..., 2] = np.roll(a[..., 2], -sp, axis=1)
    return Image.fromarray(a, "RGBA")


def saint_graph(rgb, f, acc, info, face_rel, seed):
    import hud_v5 as H5, hud_v5_kit as K
    from hud_v2 import Canvas
    pal = H5.make_pal(acc, (info or {}).get("key"))
    ck = K.Clock(f, 0.0)
    q = cv2.GaussianBlur(cv2.resize(rgb, (W // 2, H // 2), interpolation=cv2.INTER_AREA), (0, 0), 1.6)   # "背景的高斯模糊效果可以轻一些，背景也要有比较清晰的内容"
    bg = cv2.resize(q, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) * 0.50
    if "glow" not in _BG:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _BG["glow"] = np.clip(1 - np.hypot((xx - W * 0.66) / (W * 0.42), (yy - H * 0.5) / (H * 0.66)), 0, 1)[..., None] ** 1.7
    bg = bg + _BG["glow"] * np.array(pal["main"], np.float32) * 0.40 * ck.p(0, 20)
    o = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
    fr = _frame((info or {}).get("key"))
    ch = int(H * 1.12)                                      # 用户两次说"卡牌边框可以大一点"：高过画面，顶对齐（顶饰完整，底边略出画）
    if fr is not None:
        fim, (bx0, by0, bx1, by1) = fr
        sc = ch / fim.height
        cw = int(fim.width * sc)
        wx0, wy0, wx1, wy1 = [int(v * sc) for v in (bx0, by0, bx1, by1)]
        pad = int(6 * R)
        ww, wh = wx1 - wx0 + 2 * pad, wy1 - wy0 + 2 * pad
        fx = (face_rel[0] if face_rel else 0.5) * W; fy = (face_rel[1] if face_rel else 0.42) * H
        sh_ = H * 0.90; sw_ = sh_ * ww / wh
        if sw_ > W:
            sw_ = W; sh_ = sw_ * wh / ww
        sx0 = float(np.clip(fx - sw_ / 2, 0, W - sw_)); sy0 = float(np.clip(fy - sh_ * 0.40, 0, H - sh_))
        Mx = np.float32([[ww / sw_, 0, -sx0 * ww / sw_], [0, wh / sh_, -sy0 * wh / sh_]])
        por = cv2.warpAffine(rgb, Mx, (ww, wh), flags=cv2.INTER_AREA)
        card = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        card.paste(Image.fromarray(por).convert("RGBA"), (wx0 - pad, wy0 - pad))
        fl = _frame_style((cw, ch), (info or {}).get("key"), fim, pal)  # 风格化卡框（用户："展示原边框很容易暴露边框素材的缺陷"）
        ks = ck.raw(12, 26)                                 # 金属面上的一道高光扫过
        if 0 < ks < 1:
            g = np.zeros((ch, cw), np.float32)
            yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
            d = (xx + yy * 0.45) - (ks * 1.8 - 0.4) * (cw + ch * 0.45)
            g = np.exp(-(d / (40 * R)) ** 2) * 0.75
            fa = np.asarray(fl).astype(np.float32)
            fa[..., :3] = np.clip(fa[..., :3] + g[..., None] * 255 * (fa[..., 3:4] / 255), 0, 255)
            fl = Image.fromarray(fa.astype(np.uint8), "RGBA")
        card.alpha_composite(fl)
    else:
        cw = int(ch * 0.64)
        card = Image.fromarray(cv2.resize(rgb, (cw, ch))).convert("RGBA")
    kf = K.back_out(min(1.0, f / 14.0), 1.4) if f < 14 else 1.0          # 翻入：横向 0.55 → 1，回弹；上浮 + 渐显
    sx = max(0.05, 0.55 + 0.45 * kf)
    card = _card_glitch(card, f, seed)                      # 切片错位 + RGB 分离：入场时拼合，之后每 ~0.6 s 一次短 glitch
    im = card.resize((max(1, int(cw * sx)), ch), Image.BICUBIC)
    a_ = K.cl(f / 8.0)
    if a_ < 1:
        im.putalpha(im.getchannel("A").point(lambda v: int(v * a_)))
    cx = W * 0.67
    cy = ch / 2 + 4 * R + (1 - K.expo_out(f / 16.0)) * 60 * R - 4 * R * math.sin(f / 22.0)
    shd = Image.new("RGBA", (im.width + int(80 * R), im.height + int(80 * R)), (0, 0, 0, 0))
    ImageDraw.Draw(shd).rectangle([int(40 * R), int(40 * R), int(40 * R) + im.width, int(40 * R) + im.height], fill=(0, 0, 0, int(140 * a_)))
    shd = shd.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(18 * R))
    o.alpha_composite(shd, (int(cx - im.width / 2 - 30 * R), int(cy - im.height / 2 - 20 * R)))
    o.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))
    if info:                                               # 左侧信息块（HUD v5 字阶 + 动效）
        cv = Canvas()
        x0 = K.M * 1.3; y = H * 0.24
        ink = pal["ink"]
        kl = ck.p(6, 12)
        K.grow_line(cv, (x0, y), (x0 + 220 * R, y), kl, pal["main"], 1.0, 2.5 * R)
        K.ttext(cv, (x0, y + 14 * R), "SAINT GRAPH", K.fnt("label_b", 18 * R), ink, 0.8 * kl, 6 * R, decode=ck.raw(6, 12), seed=f)
        if info.get("cls"):
            K.ttext(cv, (x0 - 6 * R, y + 40 * R), info["cls"], K.fnt("display", 150 * R), ink, 1.0, 4 * R, rise=ck.raw(9, 14))
        K.ttext(cv, (x0, y + 214 * R), info["en"], K.fnt("h1", 56 * R), ink, ck.p(12, 4), 2 * R, decode=ck.raw(12, 16), seed=seed + f)
        if info.get("jp"):
            K.ttext(cv, (x0, y + 278 * R), info["jp"], K.fnt("jp", 34 * R), ink, 0.95, 3 * R, rise=ck.raw(16, 16))
        if info.get("p"):
            for j, key in enumerate(("STR", "END", "AGI", "MAN", "LUK", "NP")):
                kx = ck.p(20 + j, 6)
                K.ttext(cv, (x0 + (j % 3) * 150 * R, y + 344 * R + (j // 3) * 34 * R), "%-4s %s" % (key, info["p"].get(key, "")),
                        K.fnt("micro", 20 * R), ink, 0.7 * kx, 2 * R)
        if info.get("np_jp"):
            ny = y + 440 * R
            kn = ck.p(26, 10)
            K.ttext(cv, (x0, ny), "NOBLE PHANTASM", K.fnt("label", 16 * R), pal["hi"], kn, 5 * R)
            if info.get("np_ruby"):
                K.ttext(cv, (x0, ny + 30 * R), info["np_ruby"], K.fnt("jp_m", 20 * R), ink, 0.75 * kn, 3 * R)
            fs = 54 * R
            while fs > 26 * R and K.text_w(info["np_jp"], K.fnt("np", fs), R) > W * 0.42:
                fs -= 2 * R
            K.ttext(cv, (x0, ny + 58 * R), info["np_jp"], K.fnt("np", fs), ink, 1.0, R, rise=ck.raw(28, 16))
            meta = "  ·  ".join(x for x in (("RANK " + info["np_rank"]) if info.get("np_rank") else "", info.get("np_type", "")) if x)
            K.ttext(cv, (x0, ny + 58 * R + fs * 1.75), meta, K.fnt("label", 16 * R), ink, 0.6 * kn, 4 * R)
        o = Image.fromarray(cv.over(np.asarray(o.convert("RGB")))).convert("RGBA")
    return np.ascontiguousarray(np.asarray(o.convert("RGB")))


# ---------------------------------------------------------------- slit：画幅收窄 + 强调字
def slit_band(f, pre, n):
    """返回缝的半高（像素）。pre 帧内 A 合拢成窄缝（expo_in）；切点后缝停 40%，再弹开（expo_out）。"""
    full, narrow = H / 2.0, H * 0.115
    if f < pre:
        return full - (full - narrow) * expo_in((f + 1) / float(pre))
    g = (f - pre) / float(max(1, n - pre))
    return narrow if g < 0.40 else narrow + (full - narrow) * expo_out((g - 0.40) / 0.60)


def slit(A, B, f, n, pre, acc, text=None, sub=None):
    k = expo_in((f + 1) / float(pre)) if f < pre else 1.0
    src = A if f < pre else B
    if f < pre:                                            # A 轻推近
        s = 1.0 + 0.05 * k
        src = cv2.warpAffine(src, np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]]), (W, H), flags=cv2.INTER_LINEAR)
    else:                                                  # B 从 1.06 落回
        g = (f - pre) / float(max(1, n - pre))
        s = 1.06 - 0.06 * expo_out(g)
        src = cv2.warpAffine(src, np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]]), (W, H), flags=cv2.INTER_LINEAR)
    hb = slit_band(f, pre, n)
    y0, y1 = int(round(H / 2 - hb)), int(round(H / 2 + hb))
    out = src.copy()
    out[:max(0, y0)] = (6, 5, 6); out[min(H, y1):] = (6, 5, 6)
    lt = max(2, int(2.5 * R))
    if 0 < y0 < H // 2 - 2:                                # 缝口两条强调色细线
        out[max(0, y0 - lt):y0] = acc["main"]; out[y1:min(H, y1 + lt)] = acc["main"]
    if text and f >= pre - 2:
        g = (f - pre + 2) / float(max(1, n - pre + 2))
        a_ = 1.0 - cl((g - 0.55) / 0.30)
        if a_ > 0.02:
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
            fs = font(SF.F_BODONI_BLK, 150 * R)
            track = 6 * R + 16 * R * expo_out(g * 1.6)      # 字距慢慢拉开
            ws = [d.textlength(ch, font=fs) for ch in text]
            tw = sum(ws) + track * (len(text) - 1)
            x = W / 2 - tw / 2; y = H / 2 - 98 * R
            for ch, w_ in zip(text, ws):
                d.text((x, y), ch, font=fs, fill=(246, 244, 238, int(255 * a_)))
                x += w_ + track
            if sub:
                fsub = SF.vfont(SF.F_SERIF_CJK, int(26 * R), "Bold")     # 含日文名：用明朝（宽体没有假名）
                d.text((W / 2 - d.textlength(sub, font=fsub) / 2, H / 2 + 92 * R), sub, font=fsub, fill=tuple(acc["hi"]) + (int(230 * a_),))
            o = Image.fromarray(out).convert("RGBA"); o.alpha_composite(lay)
            out = np.asarray(o.convert("RGB"))
    return np.ascontiguousarray(out)

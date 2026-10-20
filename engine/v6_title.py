# -*- coding: utf-8 -*-
"""v9 成片（edl_v11）新增的两层：
  · caption_layer：前奏咏唱 / 间奏招式名的字幕（用户 10-20："intro 可以不放歌先做点简短介绍"）。
    serif 行：明朝体白字，下三分之一居中，字距随时间微微拉开，左侧一条角色色细线；8 帧淡入、10 帧淡出。
    burst 行：大号无衬线压缩体，3 帧冲入（1.25→1）+ 色散两层，用于原声里的招式喊名。
  · title_card：歌名 / 标题卡（召唤阵俯拍做底）："FATE SERIES  //  AMV" → PROVANT（逐字从中缝弹出、字距收拢）→ 演唱者；
    一道高光扫过；最后 8 帧整体放大 + 切片 glitch 收进第一镜。
"""
import math, os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import style_frames as SF
from style_frames import W, H, R, font
from cards import cl

F_DISP = os.path.join(r"D:\A\provant", "assets", "fonts", "SairaExtraCondensed-Black.ttf")
F_LAB = os.path.join(r"D:\A\provant", "assets", "fonts", "ChakraPetch-Bold.ttf")


def eo(x):
    x = cl(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def _fps():
    return 60.0


def caption_layer(caps, F, acc):
    """caps：[{t0, t1, text, style}]（成片秒）。返回 RGBA PIL 层或 None。"""
    t = F / _fps()
    act = [c for c in caps if c["t0"] <= t < c["t1"]]
    if not act:
        return None
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    lines = [c for c in act if c.get("style") == "line"]
    if lines:                                               # 台词演出：单独一层，压柔和暗影后再叠
        ll = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for c in lines:
            _line(ll, c, t)
        lay.alpha_composite(_soft_shadow(ll))
    for c in act:
        if c.get("style") == "line":
            continue
        f = (t - c["t0"]) * _fps(); n = (c["t1"] - c["t0"]) * _fps()
        a = min(cl(f / 8.0), cl((n - f) / 10.0))
        if a <= 0.01:
            continue
        if c.get("style") == "chapter":
            _chapter(d, c, f, n, acc)
            continue
        if c.get("style") == "player":
            _player(d, c, t, acc)
            continue
        if c.get("style", "serif") == "serif":
            fs = SF.vfont(SF.F_SERIF_CJK, int(54 * R), "Bold")
            tr = (4 + 6 * eo(f / max(1.0, n))) * R
            ws = [d.textlength(ch, font=fs) for ch in c["text"]]
            tw = sum(ws) + tr * (len(ws) - 1)
            x = W / 2 - tw / 2; y = H * 0.80
            col = tuple(int(v) for v in acc["main"])
            d.line([(x - 40 * R, y + 30 * R), (x - 14 * R, y + 30 * R)], fill=col + (int(230 * a),), width=max(2, int(3 * R)))
            for ch, w_ in zip(c["text"], ws):
                d.text((x, y), ch, font=fs, fill=(246, 244, 238, int(255 * a)))
                x += w_ + tr
        else:                                               # burst：招式喊名
            k = eo(f / 3.0); sc = 1.25 - 0.25 * k
            fs = font(F_DISP, int(190 * R * sc))
            tw = d.textlength(c["text"], font=fs)
            x = W / 2 - tw / 2; y = H * 0.40 - 95 * R * sc
            sp = int(10 * R * (1 - k) + 3 * R)
            for col, dx in ((acc["main"], -sp), (acc["sec"], sp)):
                d.text((x + dx, y), c["text"], font=fs, fill=tuple(int(v) for v in col) + (int(200 * a),))
            d.text((x, y), c["text"], font=fs, fill=(246, 244, 238, int(255 * a)), stroke_width=int(3 * R), stroke_fill=(10, 10, 12, int(255 * a)))
    return lay


def _chapter(d, c, f, n, acc):
    """间奏章节标题（左下）：色线生长 → kicker 解码 → 大标题上滑 → 副标题；末 12 帧整体左移淡出。"""
    col = tuple(int(v) for v in acc["main"])
    a = cl((n - f) / 12.0)
    dx = -(1 - a) * 80 * R
    x0, y0 = W * 0.055 + dx, H * 0.63
    kl = eo(f / 10.0)
    d.rectangle([x0, y0, x0 + 300 * R * kl, y0 + 4 * R], fill=col + (int(255 * a),))
    fk = font(F_LAB, int(24 * R))
    kt = c["kicker"]; m = int(len(kt) * cl((f - 4) / 12.0))
    d.text((x0, y0 + 14 * R), kt[:m], font=fk, fill=col + (int(255 * a),))
    gt = cl((f - 8) / 10.0)
    if gt > 0:
        ft = font(F_DISP, int(132 * R))
        d.text((x0 - 4 * R, y0 + 44 * R + (1 - eo(gt)) * 40 * R), c["text"], font=ft, fill=(246, 244, 238, int(255 * eo(gt) * a)),
               stroke_width=int(2 * R), stroke_fill=(10, 10, 12, int(200 * eo(gt) * a)))
    gs = cl((f - 16) / 10.0)
    if gs > 0 and c.get("sub"):
        fs = SF.vfont(SF.F_JP, int(30 * R), "Bold")
        d.text((x0, y0 + 196 * R), c["sub"], font=fs, fill=(230, 230, 236, int(235 * gs * a)))


def _player(d, c, t, acc):
    """break 期间的"歌曲暂停"小播放器（左下角）：暂停时 ❚❚ + 进度停在 song_t / song_len + 均衡条压平；
    恢复（t >= resume）后变 ▶、均衡条跳起来，0.9 s 后滑出。"""
    col = tuple(int(v) for v in acc["main"])
    f_in = (t - c["t0"]) * 60.0
    playing = t >= c["resume"]
    f_out = (t - c["resume"]) * 60.0 - 54 if playing else -1
    k_in = eo(f_in / 10.0); k_out = eo(f_out / 12.0) if f_out > 0 else 0.0
    a = k_in * (1 - k_out)
    if a <= 0.01:
        return
    x0 = W * 0.055 - (1 - k_in) * 60 * R - k_out * 60 * R; y0 = H * 0.885
    w_, h_ = 560 * R, 62 * R
    d.rectangle([x0, y0, x0 + w_, y0 + h_], fill=(10, 10, 12, int(170 * a)), outline=col + (int(200 * a),), width=max(1, int(1.5 * R)))
    ix, iy = x0 + 22 * R, y0 + h_ / 2
    if playing:
        d.polygon([(ix, iy - 13 * R), (ix, iy + 13 * R), (ix + 22 * R, iy)], fill=col + (int(255 * a),))
    else:
        d.rectangle([ix, iy - 13 * R, ix + 7 * R, iy + 13 * R], fill=col + (int(255 * a),))
        d.rectangle([ix + 14 * R, iy - 13 * R, ix + 21 * R, iy + 13 * R], fill=col + (int(255 * a),))
    fl = font(F_LAB, int(22 * R)); fm = font(F_LAB, int(16 * R))
    d.text((ix + 44 * R, y0 + 9 * R), c["title"], font=fl, fill=(240, 240, 244, int(255 * a)))
    st = c["song_t"] + (t - c["resume"] if playing else 0.0)
    tc = "%02d:%02d / %02d:%02d" % (int(st) // 60, int(st) % 60, int(c["song_len"]) // 60, int(c["song_len"]) % 60)
    d.text((ix + 44 * R, y0 + 37 * R), (c.get("label_play") if playing else c.get("label_pause", "")) + "   " + tc, font=fm,
           fill=(200, 200, 210, int(230 * a)))
    px0, px1, py = ix + 44 * R, x0 + w_ - 150 * R, y0 + h_ - 6 * R
    d.rectangle([px0, py, px1, py + 2 * R], fill=(80, 80, 90, int(200 * a)))
    d.rectangle([px0, py, px0 + (px1 - px0) * st / c["song_len"], py + 2 * R], fill=col + (int(255 * a),))
    bx = x0 + w_ - 128 * R
    for j in range(10):                                    # 均衡条：暂停时压平（只剩 2 px 的底噪抖动），播放时跳动
        hh = (2 + 1.5 * math.sin(t * 7 + j)) * R if not playing else (8 + 18 * abs(math.sin(t * 9.3 + j * 1.7))) * R
        d.rectangle([bx + j * 11 * R, iy + 14 * R - hh, bx + j * 11 * R + 6 * R, iy + 14 * R], fill=col + (int(220 * a),))


def title_card(base, f, n, acc, spec):
    """base = 已调色的底图（RGB）；spec = {kicker, title, sub}。"""
    fi = spec.get("fade_in", 0)                             # v10：同一镜头速度斜坡进标题卡时，暗下去要渐变（不是一刀切暗）
    dim = (0.55 - 0.15 * cl(f / 30.0)) if not fi else 1.0 - (1.0 - 0.55) * eo(f / float(fi)) - 0.15 * cl((f - fi) / 30.0)
    img = base.astype(np.float32) * dim
    s = 1.0 + 0.03 * f / max(1.0, n)                       # 缓推
    img = cv2.warpAffine(img, np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]]), (W, H), flags=cv2.INTER_LINEAR)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    col = tuple(int(v) for v in acc["main"])
    a_out = cl((n - 8 - f) / 6.0) if f > n - 14 else 1.0
    f = f - spec.get("delay", 0)                            # 文字比画面晚 delay 帧开始（先暗下来再浮字）
    fk = font(F_LAB, int(26 * R))                           # kicker：解码式逐字出现
    kt = spec["kicker"]; m = int(len(kt) * cl((f - 4) / 14.0))
    d.text((W / 2 - d.textlength(kt, font=fk) / 2, H / 2 - 170 * R), kt[:m], font=fk, fill=col + (int(255 * a_out),))
    ft = font(SF.F_BODONI_BLK, int(210 * R))               # 标题：逐字从中缝弹出 + 字距收拢
    title = spec["title"]
    ws = [d.textlength(ch, font=ft) for ch in title]
    g = cl((f - 10) / 26.0); tr = (60 - 50 * eo(g)) * R
    tw = sum(ws) + tr * (len(title) - 1); x = W / 2 - tw / 2
    for i, (ch, w_) in enumerate(zip(title, ws)):
        gi = cl((f - 10 - i * 2.2) / 7.0)
        if gi <= 0:
            x += w_ + tr; continue
        yo = (1 - eo(gi)) * 60 * R
        d.text((x, H / 2 - 125 * R + yo), ch, font=ft, fill=(246, 244, 238, int(255 * eo(gi) * a_out)))
        x += w_ + tr
    ls = cl((f - 22) / 12.0)                                # 标题下的一条角色色线
    if ls > 0:
        d.rectangle([W / 2 - 360 * R * eo(ls), H / 2 + 112 * R, W / 2 + 360 * R * eo(ls), H / 2 + 116 * R], fill=col + (int(255 * a_out),))
    fs = font(F_LAB, int(34 * R))
    sa = cl((f - 28) / 10.0) * a_out
    if sa > 0:
        d.text((W / 2 - d.textlength(spec["sub"], font=fs) / 2, H / 2 + 136 * R), spec["sub"], font=fs, fill=(230, 230, 236, int(255 * sa)))
    kl = cl((f - 30) / 24.0)                                # 高光扫过标题
    L = np.asarray(lay).astype(np.float32)
    if 0 < kl < 1:
        xx = np.arange(W, dtype=np.float32)[None, :]
        band = np.exp(-((xx - (kl * 1.4 - 0.2) * W) / (70 * R)) ** 2)
        L[..., :3] = np.clip(L[..., :3] + band[..., None] * 120 * (L[..., 3:4] / 255), 0, 255)
    al = L[..., 3:4] / 255.0
    img = img * (1 - al) + L[..., :3] * al
    if f > n - 8 and spec.get("exit", True):               # 收尾：放大 + 切片错位（后面接 slit 时不做）
        k = (f - (n - 8)) / 8.0
        s2 = 1 + 0.10 * k * k
        img = cv2.warpAffine(img, np.float32([[s2, 0, W / 2 * (1 - s2)], [0, s2, H / 2 * (1 - s2)]]), (W, H), flags=cv2.INTER_LINEAR)
        rng = np.random.default_rng(int(f))
        y = 0
        while y < H:
            hh = int(rng.integers(int(8 * R), int(70 * R)))
            if rng.random() < 0.5:
                img[y:y + hh] = np.roll(img[y:y + hh], int(rng.normal(0, 120 * R * k)), axis=1)
            y += hh
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- v10：台词演出（用户 10-20："给羽蛇神的片段和小贝的片段加上台词演出，台词做精致 motion 处理"）
# style="line"：{t0, t1, rows:[[(t, 字)…]…], who, who_jp, key, side "L"/"R", chip（可选，如 MEMORY）, shout}
#   · 说话人牌：角色色短条从外侧生长 → 英文名（字距收拢）+ 日文名上浮 + 芯片；比第一个字早 9 帧
#   · 正文：明朝体，每个字在它被说出的时刻（ASR 词级时间按字数摊开）出现：10 帧内透明度 / 上浮 18 px / 角色色残影从右收拢
#   · 进度线：正文下一条暗发丝，已说出的部分用角色色走过去
#   · 喊招（shout）：粗黑体大字，每字 5 帧从 1.5 倍砸下 + 前 6 帧抖动 + 角色色描边双影
#   · 退场：最后 18 帧逐字（左→右错 1.2 帧）上浮淡出；整块下面压一层柔和的暗影（只在字的周围），亮画面上也读得清
F_JP_SANS_B = r"C:\Windows\Fonts\YuGothB.ttc"
_FC = {}


def _f(path, size, weight=None):
    k = (path, int(size), weight)
    if k not in _FC:
        _FC[k] = SF.vfont(path, int(size), weight) if weight else font(path, int(size))
    return _FC[k]


def _line(lay, c, t):
    import palette as PAL
    d = ImageDraw.Draw(lay)
    sig = c.get("col") or PAL.SIG.get(c.get("key"), ((236, 200, 120), (120, 140, 200)))
    col = tuple(int(v) for v in sig[0])
    col2 = tuple(int(v) for v in sig[1])
    shout = c.get("shout", False)
    rows = c["rows"]
    t_first = rows[0][0][0]
    fps = _fps()
    n_out = 18
    f_end = (c["t1"] - t) * fps                            # 距结束的帧数
    side = c.get("side", "L")
    xa = W * 0.075 if side == "L" else W * 0.925
    fs = _f(F_JP_SANS_B, 104 * R) if shout else _f(SF.F_SERIF_CJK, 62 * R, "Bold")
    lh = (124 if shout else 84) * R
    y0 = H * (0.60 if shout else 0.665) - (len(rows) - 1) * lh * 0.5
    # 说话人牌
    ft = (t - (t_first - 9 / fps)) * fps
    if ft > 0:
        k = eo(ft / 12.0) * cl(f_end / 14.0)
        fl = font(F_LAB, int(24 * R))
        name = c.get("who", "")
        tr = (3 + 10 * (1 - eo(ft / 16.0))) * R
        nw = sum(d.textlength(ch, font=fl) for ch in name) + tr * max(0, len(name) - 1)
        yt = y0 - 62 * R
        bar = 54 * R * eo(ft / 10.0)
        if side == "L":
            d.rectangle([xa, yt + 10 * R, xa + bar, yt + 14 * R], fill=col + (int(255 * k),))
            x = xa + 70 * R
        else:
            d.rectangle([xa - bar, yt + 10 * R, xa, yt + 14 * R], fill=col + (int(255 * k),))
            x = xa - 70 * R - nw
        xx = x
        for ch in name:
            d.text((xx, yt), ch, font=fl, fill=(246, 244, 238, int(255 * k)))
            xx += d.textlength(ch, font=fl) + tr
        fj = _f(SF.F_JP, 20 * R, "Bold")
        kj = eo((ft - 5) / 12.0) * cl(f_end / 14.0)
        if c.get("who_jp") and kj > 0:
            jx = xx + 16 * R if side == "L" else x - 16 * R - d.textlength(c["who_jp"], font=fj)
            d.text((jx, yt + 3 * R + (1 - kj) * 10 * R), c["who_jp"], font=fj, fill=col + (int(235 * kj),))
            if c.get("chip"):
                fc = font(F_LAB, int(15 * R))
                cw = d.textlength(c["chip"], font=fc) + 18 * R
                cx0 = jx + d.textlength(c["who_jp"], font=fj) + 16 * R if side == "L" else jx - 16 * R - cw
                d.rectangle([cx0, yt + 4 * R, cx0 + cw * kj, yt + 28 * R], outline=col2 + (int(230 * kj),), width=max(1, int(1.5 * R)))
                if kj > 0.8:
                    d.text((cx0 + 9 * R, yt + 6 * R), c["chip"], font=fc, fill=col2 + (int(255 * kj),))
    # 正文
    for ri, row in enumerate(rows):
        ws = [d.textlength(ch, font=fs) for _, ch in row]
        trk = 2 * R if not shout else 6 * R
        tw = sum(ws) + trk * (len(ws) - 1)
        x = xa if side == "L" else xa - tw
        y = y0 + ri * lh
        if not shout:                                       # 进度线
            ka = eo((t - row[0][0] + 6 / fps) * fps / 10.0) * cl(f_end / 12.0)
            if ka > 0:
                yl = y + 78 * R
                d.line([(x, yl), (x + tw * ka, yl)], fill=(246, 244, 238, int(70 * ka)), width=max(1, int(1.2 * R)))
                xe = x
                for (tc, ch), w_ in zip(row, ws):
                    if t >= tc:
                        xe += w_ + trk
                d.line([(x, yl), (min(x + tw, xe), yl)], fill=col + (int(230 * ka),), width=max(2, int(2.4 * R)))
        for j, ((tc, ch), w_) in enumerate(zip(row, ws)):
            f = (t - tc) * fps
            if f < 0:
                x += w_ + trk; continue
            jo = sum(len(r_) for r_ in rows[:ri]) + j      # 退场按全局字序错开
            ko = cl((f_end - jo * 1.2) / float(n_out)) if f_end < n_out + jo * 1.2 else 1.0
            if shout:
                k = eo(f / 5.0)
                sc = 1.5 - 0.5 * k
                fz = _f(F_JP_SANS_B, 104 * R * sc)
                cw_ = d.textlength(ch, font=fz)
                rng = np.random.default_rng(int(tc * 1000) + int(f))
                jx, jy = (rng.normal(0, 4 * R, 2) if f < 6 else (0.0, 0.0))
                px = x + w_ / 2 - cw_ / 2 + jx; py = y - (104 * R * sc - 104 * R) * 0.6 + jy - (1 - ko) * 14 * R
                a = int(255 * min(1.0, f / 2.0) * ko)
                for cc, dx in ((col, -5 * R * (1 - k) - 2 * R), (col2, 5 * R * (1 - k) + 2 * R)):
                    d.text((px + dx, py), ch, font=fz, fill=cc + (int(a * 0.75),))
                d.text((px, py), ch, font=fz, fill=(250, 248, 242, a), stroke_width=int(3 * R), stroke_fill=(12, 10, 14, a))
            else:
                k = eo(f / 10.0)
                yo = (1 - k) * 18 * R - (1 - ko) * 12 * R
                a = int(255 * k * ko)
                if k < 1:
                    d.text((x + (1 - k) * 14 * R, y + yo), ch, font=fs, fill=col + (int(150 * (1 - k) * ko),))
                d.text((x, y + yo), ch, font=fs, fill=(248, 246, 240, a), stroke_width=max(1, int(1.6 * R)), stroke_fill=(10, 10, 14, int(a * 0.55)))
            x += w_ + trk


def _soft_shadow(lay):
    """字周围压一层柔和暗影：alpha 膨胀 + 模糊，只处理有字的包围盒。"""
    A = np.asarray(lay)[..., 3]
    ys, xs = np.nonzero(A > 8)
    if not len(ys):
        return lay
    pad = int(60 * R)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    a = A[y0:y1, x0:x1].astype(np.float32) / 255
    a = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(17 * R), int(17 * R))))
    a = cv2.GaussianBlur(a, (0, 0), 14 * R) * 0.62
    sh = np.zeros((H, W, 4), np.uint8)
    sh[y0:y1, x0:x1, 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    sh[y0:y1, x0:x1, :3] = (8, 8, 12)
    out = Image.fromarray(sh, "RGBA")
    out.alpha_composite(lay)
    return out

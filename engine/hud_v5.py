# -*- coding: utf-8 -*-
"""
HUD v5 模板（组件 / 设计系统 / 动效约定见 hud_v5_kit.py）。每镜一个模板，1–3 个大结构 + 1 个信息块，铺开画面而不堆在一角；
文字只写核对过的信息（data/servant_info.json；查不到的只写名字 / 职阶）。
  lock     分段大环（72 段依次点亮、三段强调色、慢转）+ 内刻度环 + 外轨道弧 + 四向指针回弹；脸上括号飞入锁定闪两下；
           背后巨型描边职阶字（被人物遮挡）；远侧真名块（引线 → 名字解码 → 日文名上浮 → 职阶芯片）
  dossier  远侧竖版档案面板：框线生长 → 填充 → 标题条展开 → 职阶大字 → 真名 → 刻度分隔线 → 六项分格量表逐格点亮 + 等级字母回弹 → 宝具
  type     字体主视觉：横贯画面的巨型描边真名（强调色从左扫满，人物在字前）+ 上下两条带刻度的发丝线 + 角落信息块 + 边缘竖排日文名
  hexwall  透视蜂巢墙（远侧斜墙，近大远小，从近边一波弹出，实心格写职阶 / 宝具等级）+ 扫描线 + 顶部真名条
  np       宝具横幅：斜向双层条（强调色等级徽章 + 日文宝具名逐字落位 + 读音 + 英文名 / 种类芯片）+ 目标处放射线
  circle   召唤阵（生成的透明底素材，透视压扁、慢转，在人物脚下被人物遮挡）+ 顶部字距展开的 SERVANT 行
  scope    分段准星环 + 毫位刻度 + 间隙十字 + 切角放大窗（标题条 ZOOM）
  v10 新增（用户 10-20："HUD 种类也可以多设计和加入一些"）：
  ~~versus~~ 双人对决牌（VS / PLAYER 名牌）——用户看过 v10："有些出戏，并且这种设计有点太幼稚"，已删
  link     双人标注：各自细角标 + 外侧名字 / 职阶，两人之间虚发丝线 + 刻度 + 中点切口 + 实测距离（px）
  blueprint 技术制图：延伸线 + 箭头尺寸线（实测框宽高）+ 点划中心线 + 编号引线标注（真名 / 职阶 / 宝具）+ 图框标题栏
  codex    竖排编辑式：远侧大号明朝竖排日文真名（被人物遮挡、逐字显影）+ 双细线 + 英文名 / 职阶 / 宝具日文名 + 读音的小版式 + 引线
  track    运动轨迹：目标的真实跟踪尾迹 + 时间刻度 + 间隙十字 + 航向箭头，读数块（屏幕速度 px/s / 航向角 / 锁定时长，全由轨迹算出）
  viewfinder 摄影机取景器：半透明 2.39:1 遮幅 + 角标 + 三分线 + REC / 成片时间码 / 作品名 + 对焦括号 + 真实音乐电平表
"""
import math, os, re
import numpy as np
import cv2
from PIL import Image, ImageDraw

from style_frames import W, H, R
from cards import cl
import hud_elems as HE
import v3_hex as HX
from hud_v2 import Canvas
from hud_v3 import premult, glitch_rows
from hud_v4 import info_for, rank_v, WORK  # noqa: F401（引擎从这里取）
import hud_v5_kit as K
from hud_v5_kit import M, U, INK, DIM, HAIR, WHITE, ON_ACC, PANEL, fnt, expo_out, back_out, Clock, ttext, text_w

ROOT = r"D:\A\provant"
TEMPLATES = ("lock", "dossier", "type", "hexwall", "np", "circle", "scope", "track", "viewfinder", "link", "blueprint", "codex")
_CIRCLE = {}


# 角色签名色（主 / 副）：HUD 主色 = 签名色 × 0.62 + 画面主色 × 0.38（用户："HUD 的配色要根据画面和角色的配色来变"）
SIG = {"jeanne": ((226, 190, 96), (70, 96, 170)), "artoria": ((70, 120, 230), (232, 196, 90)), "artoria_alter": ((214, 44, 52), (150, 140, 170)),
       "okita": ((80, 204, 224), (244, 150, 186)), "okita_alter": ((226, 44, 66), (196, 186, 206)), "musashi": ((76, 136, 236), (226, 66, 76)),
       "medusa": ((160, 96, 226), (226, 80, 170)), "karna": ((244, 184, 64), (210, 60, 44)), "mordred": ((226, 56, 44), (206, 206, 216)),
       "mash": ((176, 126, 236), (244, 166, 206)), "richard": ((216, 56, 56), (244, 194, 84))}
import palette as _PAL
SIG.update({k: v for k, v in _PAL.SIG.items() if k not in SIG})      # v10 新增英灵（伊什塔尔 / 阿拉什 / 贝德维尔 / 狮子王 / 魁札尔）


def make_pal(acc, key):
    import colorsys
    fm = np.array(acc["main"], np.float32)
    if key in SIG:
        sm = np.array(SIG[key][0], np.float32)
        h1 = colorsys.rgb_to_hsv(*(sm / 255.0))[0]; h2 = colorsys.rgb_to_hsv(*(np.clip(fm, 1, 255) / 255.0))[0]
        dh = min(abs(h1 - h2), 1 - abs(h1 - h2)) * 360
        if dh <= 60:                                        # 色相接近：混合（跟着画面走）
            main = 0.62 * sm + 0.38 * fm; sub = np.array(SIG[key][1], np.float32)
        else:                                               # 色相冲突（蓝 vs 金）：主色用角色签名色，画面色做副色，避免混出脏灰
            main = sm; sub = fm
    else:
        main = fm; sub = np.array(acc.get("sec", acc["main"]), np.float32)

    def lift(c, smin=0.50, vmin=0.82):
        h, s_, v = colorsys.rgb_to_hsv(*(np.clip(c, 0, 255) / 255.0))
        return np.array(colorsys.hsv_to_rgb(h, max(s_, smin), max(v, vmin)), np.float32) * 255
    main, sub = lift(main), lift(sub, 0.35, 0.78)
    ink = np.array((240, 244, 246), np.float32) * 0.86 + (main * 0.5 + 255 * 0.5) * 0.14
    hi = main * 0.55 + 255 * 0.45
    t = lambda c: tuple(int(v) for v in np.clip(c, 0, 255))
    return dict(acc, main=t(main), sec=t(sub), hi=t(hi), ink=t(ink))


def blocks(rank):
    """等级 → 量表格数（12 格：E2 D4 C6 B8 A10 EX12，每个 + 加 1）。"""
    if not rank:
        return 0
    if rank.startswith("EX"):
        return 12
    if rank[0] not in "EDCBA":
        return 0
    return min(12, 2 * ("EDCBA".index(rank[0]) + 1) + rank.count("+"))


def short_name(inf):
    en = inf["en"].replace("[ALTER]", "").strip()
    parts = en.split()
    return parts[0] if len(parts) > 1 and len(parts[0]) >= 4 else en


def _dash(cv, p, q, col, a, th, on, off, dot=False):
    """虚线 / 点划线（dot=True：每段后加一个短点，制图中心线）。"""
    L_ = math.hypot(q[0] - p[0], q[1] - p[1])
    if L_ < 1:
        return
    ux, uy = (q[0] - p[0]) / L_, (q[1] - p[1]) / L_
    d = 0.0
    while d < L_:
        e = min(L_, d + on)
        cv.line((p[0] + ux * d, p[1] + uy * d), (p[0] + ux * e, p[1] + uy * e), col, a, th)
        d = e + off
        if dot and d < L_:
            e = min(L_, d + 2.5 * R)
            cv.line((p[0] + ux * d, p[1] + uy * d), (p[0] + ux * e, p[1] + uy * e), col, a, th)
            d = e + off


def _dim(cv, p, q, k, col, a):
    """尺寸线：从两端向中间画，两端实心箭头。"""
    L_ = math.hypot(q[0] - p[0], q[1] - p[1])
    if L_ < 1 or k <= 0:
        return
    ux, uy = (q[0] - p[0]) / L_, (q[1] - p[1]) / L_
    nx, ny = -uy, ux
    m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    for e, s_ in ((p, 1), (q, -1)):
        cv.line(e, (e[0] + (m[0] - e[0]) * k, e[1] + (m[1] - e[1]) * k), col, a, 1.2 * R)
        h, w = 14 * R, 4.5 * R
        cv.tri([e, (e[0] + s_ * ux * h + nx * w, e[1] + s_ * uy * h + ny * w), (e[0] + s_ * ux * h - nx * w, e[1] + s_ * uy * h - ny * w)], col, a * cl(k * 2))


class HudV5:
    def __init__(self, seed, kind, info, work=""):
        rng = np.random.default_rng(seed * 7 + 311)
        self.seed, self.kind, self.info, self.work = seed, kind, info, work
        self.depth = float(rng.uniform(0.75, 0.95))
        self.rot0 = float(rng.uniform(0, 360))
        self.path, self.vs, self.rms_now, self.rms_seg, self.rms_top, self.t_now = None, None, None, None, 1.0, 0.0

    def _side(self, tc):
        return "r" if tc[0] < W * 0.5 else "l"

    def _facebox(self, box, tc):
        x0, y0, x1, y1 = box
        if x1 - x0 > 0.30 * W or y1 - y0 > 0.50 * H:
            s = 0.15 * W
            return (tc[0] - s / 2, tc[1] - s * 0.42, tc[0] + s / 2, tc[1] + s * 0.42)
        return (x0, y0, x1, y1)

    # ============================================================ lock
    def t_lock(self, B, F, ck, tc, box, acc):
        cx, cy = tc
        bw, bh = box[2] - box[0], box[3] - box[1]
        r = float(np.clip(0.55 * max(bw, bh), 0.28 * H, 0.44 * H))
        rot = self.rot0 + 0.22 * ck.f
        inf = self.info or {}
        side = self._side(tc)
        # 背后巨型描边职阶字（世界层：被人物遮挡）
        word = (inf.get("cls") or "").replace(" ", "") or short_name(inf) if inf else ""
        if word:
            k = ck.p(0, 16, 0.05)
            im = K.outline_text(word, fnt("display", 0.66 * H), self.ink, 0.30 * k, 2.2 * R, fill=tuple(acc["main"]) + (int(70 * k),),
                                fill_k=expo_out((ck.f - 8) / 30.0), track=6 * R)
            x = W / 2 - im.width / 2 - 0.5 * R * ck.f + (60 * R if side == "r" else -60 * R)
            y = H * 0.50 - im.height * 0.55 + (1 - k) * 40 * R
            B.txt.alpha_composite(im, (int(x), int(y))) if im.width < W * 3 else None
        # 分段大环：72 段依次点亮
        n = 72
        acc_set = set(range(0, 6)) | set(range(24, 28)) | set(range(48, 54))
        for j in range(n):
            kj = ck.p(2 + j * 0.13, 5, 0.3)
            if kj <= 0.01:
                continue
            a0 = rot + j * 5.0 - 90
            on = j in acc_set
            K.sector(B, (cx, cy), r, r + (12 if on else 8) * R * kj, a0, a0 + 3.6, acc["main"] if on else self.ink, 1.0 if on else 0.55 * kj)
        # 内刻度环（反转）+ 发丝圆
        ki = ck.p(4, 14, 0.4)
        K.arc(B, (cx, cy), r * 0.88, -90, -90 + 360 * ki, self.ink, HAIR + 0.1, 1.2 * R)
        if ki > 0.3:
            K.ticks(B, (cx, cy), r * 0.88, 72, 16 * R * ki, 6 * R * ki, -0.4 * ck.f, self.ink, 0.6 * ki, 1.2 * R, every=6)
        # 外轨道弧（120°，两端短竖）
        ko = ck.p(8, 14, 0.5)
        a0 = -rot * 1.6 + 40
        K.arc(B, (cx, cy), r + 34 * R, a0, a0 + 120 * ko, self.ink, 0.7, 1.6 * R)
        for aa in (a0, a0 + 120 * ko):
            u = math.radians(aa)
            B.line((cx + math.cos(u) * (r + 26 * R), cy + math.sin(u) * (r + 26 * R)), (cx + math.cos(u) * (r + 42 * R), cy + math.sin(u) * (r + 42 * R)),
                   self.ink, 0.8 * ko, 1.6 * R)
        # 精密层：方位刻度字（000 / 090 / 180 / 270）+ 次环 8 段短弧（反向快转）+ 外圈点环
        kd = ck.p(12, 10, 0.45)
        if kd > 0.02:
            for q, lab in enumerate(("000", "090", "180", "270")):
                u = math.radians(-90 + 90 * q)
                ttext(B, (cx + math.cos(u) * (r + 70 * R), cy + math.sin(u) * (r + 70 * R) - 8 * R), lab, fnt("micro", 13 * R), self.ink, 0.5 * kd,
                      1 * R, align="c")
            for q in range(8):
                a0 = -1.1 * ck.f + q * 45
                K.arc(B, (cx, cy), r * 0.80, a0, a0 + 18 * kd, acc["sec"], 0.8, 2.4 * R)
            for q in range(120):
                u = math.radians(q * 3 + rot * 0.3)
                if q % 2 == 0:
                    B.circle((cx + math.cos(u) * (r + 22 * R), cy + math.sin(u) * (r + 22 * R)), 1.3 * R, self.ink, 0.45 * kd, -1)
        # 四向指针：从外飞入、回弹
        kp = ck.p(8, 12, 0.6, ease=back_out)
        for q in range(4):
            u = math.radians(45 + 90 * q)
            d = r + 52 * R + (1 - kp) * 160 * R
            p = (cx + math.cos(u) * d, cy + math.sin(u) * d)
            nx, ny = -math.cos(u), -math.sin(u); tx_, ty_ = -ny, nx
            s = 14 * R
            if kp > 0.02:
                B.tri([(p[0] + nx * s, p[1] + ny * s), (p[0] - tx_ * s * 0.7, p[1] - ty_ * s * 0.7), (p[0] + tx_ * s * 0.7, p[1] + ty_ * s * 0.7)],
                      self.ink, 0.9 * cl(kp))
        return r

    def f_lock(self, Fr, ck, tc, box, acc, r):
        cx, cy = tc
        inf = self.info or {}
        side = self._side(tc)
        fb = self._facebox(box, tc)
        kb = ck.p(4, 12, 0.7, ease=back_out)
        s = 1 + 0.9 * (1 - kb)
        bx = ((fb[0] - cx) * s + cx, (fb[1] - cy) * s + cy, (fb[2] - cx) * s + cx, (fb[3] - cy) * s + cy)
        blink = 0.25 if (16 <= ck.f < 18 or 20 <= ck.f < 22) else 1.0
        if kb > 0.02:
            K.bracket(Fr, bx, 34 * R, 3.0 * R, acc["main"] if ck.f >= 22 else self.ink, cl(kb) * blink)
        kt = ck.p(14, 10, 0.75)
        if kt > 0.01 and inf:
            lab = "LOCK"
            fl = fnt("label_b", 16 * R)
            tw = K.text_w(lab, fl, 3 * R)
            x0, y0 = bx[0], bx[1] - 34 * R
            Fr.rect((x0, y0), (x0 + (tw + 16 * R) * kt, y0 + 24 * R), acc["main"], 1.0)
            ttext(Fr, (x0 + 8 * R, y0 + 2 * R), lab, fl, ON_ACC, kt, 3 * R)
            ttext(Fr, (x0 + tw + 28 * R, y0 + 3 * R), inf["en"], fnt("micro", 16 * R), self.ink, INK * kt, 2 * R, decode=ck.raw(16, 12), seed=ck.f)
        if not inf:
            return
        # 真名块：环上一点 → 引线 → 名字
        ang = math.radians(-32 if side == "r" else 212)
        p0 = (cx + math.cos(ang) * (r + 14 * R), cy + math.sin(ang) * (r + 14 * R))
        ex = W - M - 520 * R if side == "r" else M + 520 * R
        y = float(np.clip(p0[1] - 80 * R, M + 40 * R, H - M - 260 * R))
        kl = ck.p(10, 10, 0.8)
        K.poly_grow(Fr, [p0, (p0[0] + (40 * R if side == "r" else -40 * R), y + 70 * R), (ex, y + 70 * R)], kl, self.ink, 0.8, 1.4 * R, closed=False)
        Fr.circle(p0, 5 * R, acc["main"], cl(kl * 3), -1)
        xn = ex + (16 * R if side == "r" else -16 * R)
        al = "l" if side == "r" else "r"
        ttext(Fr, (xn, y - 14 * R), inf["en"], fnt("h1", 64 * R), self.ink, INK * ck.p(12, 6, 0.85), 2 * R, align=al, decode=ck.raw(12, 16), seed=self.seed + ck.f)
        if inf.get("jp"):
            ttext(Fr, (xn, y + 86 * R), inf["jp"], fnt("jp", 34 * R), self.ink, INK, 2 * R, align=al, rise=ck.raw(18, 16) * ck.p(0, 1, 0.9))
        if inf.get("cls"):
            fl = fnt("label_b", 20 * R)
            tw = K.text_w(inf["cls"], fl, 4 * R)
            kc = ck.p(22, 8, 0.95)
            cx0 = xn if side == "r" else xn - tw - 28 * R
            HE.fpoly(Fr, K.notch_rect(cx0, y + 146 * R, cx0 + (tw + 28 * R) * kc, y + 180 * R, 8 * R), acc["main"], 1.0)
            if kc > 0.9:
                ttext(Fr, (cx0 + 14 * R, y + 149 * R), inf["cls"], fl, ON_ACC, 1.0, 4 * R)
            if inf.get("p"):                                # 微型参数表（真实数据，两行三列）
                fm = fnt("micro", 15 * R)
                for j, key in enumerate(("STR", "END", "AGI", "MAN", "LUK", "NP")):
                    kx = ck.p(26 + j, 6, 0.97)
                    col_, row_ = j % 3, j // 3
                    xx = (cx0 if side == "r" else cx0 + tw + 28 * R - 3 * 110 * R) + col_ * 110 * R
                    ttext(Fr, (xx, y + 196 * R + row_ * 24 * R), "%-4s%s" % (key, inf["p"].get(key, "")), fm, self.ink, 0.62 * kx, 2 * R)
            if inf.get("np_rank") and inf["np_rank"] not in ("—", ""):
                ttext(Fr, (cx0 + tw + 44 * R if side == "r" else cx0 - 16 * R, y + 152 * R), "NOBLE PHANTASM  " + inf["np_rank"],
                      fnt("label", 16 * R), self.ink, DIM * kc + 0.2, 3 * R, align="l" if side == "r" else "r", decode=ck.raw(24, 10), seed=ck.f)

    # ============================================================ dossier
    def t_dossier(self, B, F, ck, tc, box, acc):
        cx, cy = tc
        fb = self._facebox(box, tc)
        kb = ck.p(3, 10, 0.2, ease=back_out)
        if kb > 0.02:
            s = 1 + 0.6 * (1 - kb)
            m = ((fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2)
            K.bracket(B, tuple(m[i % 2] + (fb[i] - m[i % 2]) * s for i in range(4)), 26 * R, 2.4 * R, self.ink, cl(kb))
        return None

    def f_dossier(self, Fr, ck, tc, box, acc, _):
        inf = self.info
        side = self._side(tc)
        pw = 0.285 * W
        x0 = W - M - pw if side == "r" else M
        x1 = x0 + pw
        y0, y1 = M * 0.9, H - M * 0.9
        fb = self._facebox(box, tc)
        # 引线：括号 → 面板
        kl = ck.p(6, 12, 0.1)
        ax = fb[2] if side == "r" else fb[0]
        K.poly_grow(Fr, [(ax, fb[1]), (ax + (60 * R if side == "r" else -60 * R), y0 + 120 * R), (x0 if side == "r" else x1, y0 + 120 * R)],
                    kl, self.ink, 0.7, 1.4 * R, closed=False)
        # 面板：框线生长 → 填充 → 标题条
        outline = K.notch_rect(x0, y0, x1, y1, 22 * R, (1, 0, 1, 0))
        K.poly_grow(Fr, outline, ck.p(2, 14, 0.15), self.ink, 0.75, 1.4 * R)
        kf = ck.p(10, 10, 0.2)
        if kf > 0.01:
            HE.fpoly(Fr, outline, PANEL, 0.52 * kf)
            g_ = int(24 * R)                                # 面板点阵
            xs_, ys_ = np.arange(int(x0) + g_, int(x1) - 4, g_), np.arange(int(y0) + 60 * int(R), int(y1) - 4, g_)
            ink4 = np.array(tuple(v * 0.16 * kf for v in self.ink) + (255 * 0.16 * kf,), np.float32)
            for yy in ys_:
                Fr.a[yy, xs_] = np.maximum(Fr.a[yy, xs_], ink4.astype(np.uint8))
        kh = ck.p(8, 12, 0.3)
        hx1 = x0 + 22 * R + (pw - 22 * R) * kh
        if kh > 0.01:
            Fr.rect((x0 + 22 * R, y0), (hx1, y0 + 44 * R), acc["main"], 1.0)
            if kh > 0.6:
                K.stripes(Fr, x1 - 150 * R, y0 + 6 * R, min(hx1, x1) - 10 * R, y0 + 38 * R, ON_ACC, 0.55)
                ttext(Fr, (x0 + 40 * R, y0 + 8 * R), "SERVANT  FILE", fnt("label_b", 22 * R), ON_ACC, 1.0, 5 * R, decode=ck.raw(14, 10), seed=ck.f)
        px = x0 + 40 * R
        if inf.get("cls"):
            ttext(Fr, (px - 4 * R, y0 + 58 * R), inf["cls"], fnt("display", 150 * R), self.ink, INK, 4 * R, rise=ck.raw(12, 12) * ck.p(0, 1, 0.4))
        ttext(Fr, (px, y0 + 228 * R), inf["en"], fnt("h1", 46 * R), self.ink, INK * ck.p(14, 4, 0.45), 2 * R, decode=ck.raw(14, 14), seed=self.seed + ck.f)
        if inf.get("jp"):
            ttext(Fr, (px, y0 + 284 * R), inf["jp"], fnt("jp", 30 * R), self.ink, INK, 2 * R, rise=ck.raw(18, 14) * ck.p(0, 1, 0.5))
        yd = y0 + 346 * R
        kd = ck.p(18, 12, 0.5)
        K.grow_line(Fr, (px, yd), (x1 - 40 * R, yd), kd, self.ink, 0.5, 1.2 * R)
        for t in range(0, int((x1 - 40 * R - px) * kd), int(16 * R)):
            Fr.line((px + t, yd), (px + t, yd + (10 if (t // int(16 * R)) % 5 == 0 else 5) * R), self.ink, 0.4, 1.0 * R)
        if inf.get("p"):
            fl, fr_ = fnt("label", 20 * R), fnt("display", 50 * R)
            mx0, mx1 = px + 92 * R, x1 - 120 * R
            for j, key in enumerate(("STR", "END", "AGI", "MAN", "LUK", "NP")):
                ry = yd + 30 * R + j * 52 * R
                t0 = 20 + 2 * j
                kr = ck.p(t0, 6, 0.6 + 0.03 * j)
                if kr <= 0.01:
                    continue
                ttext(Fr, (px, ry + 4 * R), key, fl, self.ink, DIM + 0.35 * kr, 4 * R)
                rk = inf["p"].get(key, "")
                nb = blocks(rk)
                lit = min(nb, max(0.0, (ck.f - t0 - 2) / 1.4)) * ck.p(0, 1, 0.6 + 0.03 * j)
                K.seg_meter(Fr, mx0, ry + 10 * R, mx1 - mx0, 20 * R, 12, lit, acc["main"], kr)
                kk = ck.p(t0 + 2 + nb * 1.4, 8, 0.62 + 0.03 * j, ease=back_out)
                if kk > 0.01:
                    ttext(Fr, (x1 - 40 * R, ry - 14 * R - (1 - cl(kk)) * 10 * R), rk, fr_, self.ink, INK * cl(kk), 1 * R, align="r")
        if inf.get("np_jp"):
            ny = y1 - 186 * R
            kn = ck.p(34, 10, 0.9)
            ttext(Fr, (px, ny), "NOBLE PHANTASM", fnt("label", 16 * R), acc["hi"], kn, 4 * R, decode=ck.raw(34, 10), seed=ck.f)
            if inf.get("np_ruby"):
                ttext(Fr, (px, ny + 30 * R), inf["np_ruby"], fnt("jp_m", 18 * R), self.ink, DIM + 0.3 * kn, 3 * R)
            fs_ = 44 * R
            while fs_ > 22 * R and K.text_w(inf["np_jp"], fnt("np", fs_), 1 * R) > x1 - 40 * R - px:      # 长宝具名自动缩
                fs_ -= 2 * R
            ttext(Fr, (px, ny + 56 * R), inf["np_jp"], fnt("np", fs_), self.ink, INK, 1 * R, rise=ck.raw(36, 16) * ck.p(0, 1, 0.92))
            meta = "  ·  ".join(x for x in (("RANK " + inf["np_rank"]) if inf.get("np_rank") else "", inf.get("np_type", "")) if x)
            ttext(Fr, (px, ny + 120 * R), meta, fnt("label", 15 * R), self.ink, DIM * kn + 0.25 * kn, 3 * R)

    # ============================================================ type
    def t_type(self, B, F, ck, tc, box, acc):
        inf = self.info or {}
        word = short_name(inf) if inf else ""
        if not word:
            return None
        k = ck.p(0, 14, 0.1)
        f_ = fnt("display", 0.62 * H)
        im = K.outline_text(word, f_, self.ink, 0.80 * k, 3.0 * R, fill=tuple(acc["main"]) + (int(150 * k),), fill_k=expo_out((ck.f - 6) / 26.0), track=10 * R)
        if im.width > 1.15 * W:
            im = im.resize((int(1.15 * W), int(im.height * 1.15 * W / im.width)), Image.LANCZOS)
        x = W / 2 - im.width / 2 + (1 - k) * 120 * R - 0.35 * R * ck.f
        y = H * 0.52 - im.height / 2
        B.txt.alpha_composite(im, (int(x), int(y)))
        kl = ck.p(2, 14, 0.2)
        for yy in (y + im.height * 0.10, y + im.height * 0.92):
            K.grow_line(B, (0, yy), (W, yy), kl, self.ink, 0.45, 1.2 * R, from_mid=True)
            for t in range(int(W / 2 - W / 2 * kl), int(W / 2 + W / 2 * kl), int(40 * R)):
                B.line((t, yy), (t, yy + (12 if (t // int(40 * R)) % 4 == 0 else 5) * R), self.ink, 0.4, 1.0 * R)
        return None

    def f_type(self, Fr, ck, tc, box, acc, _):
        inf = self.info
        if not inf:
            return
        side = self._side(tc)
        x0, y0 = (M, H - M - 120 * R)
        kc = ck.p(10, 10, 0.6)
        fl = fnt("label_b", 20 * R)
        if inf.get("cls"):
            tw = K.text_w(inf["cls"], fl, 4 * R)
            HE.fpoly(Fr, K.notch_rect(x0, y0, x0 + (tw + 28 * R) * kc, y0 + 34 * R, 8 * R), acc["main"], 1.0)
            if kc > 0.9:
                ttext(Fr, (x0 + 14 * R, y0 + 3 * R), inf["cls"], fl, ON_ACC, 1.0, 4 * R)
        ttext(Fr, (x0, y0 + 48 * R), inf["en"], fnt("h2", 34 * R), self.ink, INK * ck.p(12, 4, 0.7), 3 * R, decode=ck.raw(12, 14), seed=ck.f)
        if self.work:
            ttext(Fr, (x0, y0 - 34 * R), self.work, fnt("label", 15 * R), self.ink, DIM * ck.p(8, 8, 0.5), 4 * R)
        if inf.get("jp"):                                   # 边缘竖排日文名 + 细线
            xv = W - M * 0.6 - 40 * R if side == "r" else M * 0.6
            kv = ck.p(14, 18, 0.8)
            L_ = K.vtext(Fr, xv, M, inf["jp"], fnt("jp", 40 * R), self.ink, INK * ck.p(0, 1, 0.8), k=ck.raw(14, 18))
            K.grow_line(Fr, (xv + 20 * R, M + L_ + 20 * R), (xv + 20 * R, M + L_ + 220 * R), kv, acc["main"], 1.0, 2 * R)

    # ============================================================ hexwall
    def t_hexwall(self, B, F, ck, tc, box, acc):
        side = self._side(tc)
        if side == "r":
            quad = np.float32([[0.50 * W, 0.08 * H], [1.10 * W, -0.16 * H], [1.10 * W, 1.16 * H], [0.50 * W, 0.92 * H]])
        else:
            quad = np.float32([[-0.10 * W, -0.16 * H], [0.50 * W, 0.08 * H], [0.50 * W, 0.92 * H], [-0.10 * W, 1.16 * H]])
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), quad)
        rp = 0.052
        dx, dy = 1.5 * rp, math.sqrt(3) * rp * 1.15
        inf = self.info or {}
        words = [w for w in (inf.get("cls", ""), ("NP " + inf["np_rank"]) if inf.get("np_rank") and inf["np_rank"] != "—" else "",
                             "宝具" if inf.get("np_jp") else "", "真名" if inf.get("jp") else "") if w]
        rng = np.random.default_rng(self.seed)
        cells = []
        q = 0; u = 0.0
        while u < 1.0 + rp:
            for kk in range(int(1 / dy) + 2):
                v = kk * dy + (q % 2) * dy / 2
                near = min(1.0, max(0.0, u if side == "r" else 1 - u))
                if rng.random() < 0.25 + 0.70 * near ** 0.7:
                    cells.append((u, v, near))
            u += dx; q += 1
        filled = set(int(i) for i in rng.choice(len(cells), min(len(cells), max(3, len(words) + 2)), replace=False))
        lab = {c_: w for c_, w in zip(sorted(filled), words)}
        for idx, (u, v, near) in enumerate(cells):
            t0 = 1 + (1 - near) * 22 + rng.random() * 3
            kc = ck.p(t0, 6, 0.1 + 0.6 * (1 - near), ease=back_out)
            if kc <= 0.02:
                continue
            sc = 0.35 + 0.65 * cl(kc)
            pts = [(u + rp * 0.92 * sc * math.cos(math.radians(60 * j)), v + rp * 0.92 * 1.15 * sc * math.sin(math.radians(60 * j))) for j in range(6)]
            sp = cv2.perspectiveTransform(np.float32(pts)[None], Hm)[0].tolist()
            if idx in filled:
                HE.fpoly(B, sp, acc["main"], 0.95)
                w_ = lab.get(idx)
                if w_ and kc > 0.9:
                    c = cv2.perspectiveTransform(np.float32([[[u, v]]]), Hm)[0, 0]
                    size = abs(sp[0][0] - sp[3][0]) * (0.34 if len(w_) <= 2 else 0.22)
                    f_ = fnt("jp", size) if ord(w_[0]) > 0x2E80 else fnt("display", size * 1.25)
                    tw = K.text_w(w_, f_)
                    B.d.text((c[0] - tw / 2, c[1] - size * 0.72), w_, font=f_, fill=ON_ACC + (255,))
            else:
                HE.poly(B, sp, self.ink, 0.30 + 0.35 * near, (1.0 + 0.8 * near) * R)
        ks = ck.raw(16, 24)                                  # 扫描线：沿墙面从近边扫到远边
        if 0 < ks < 1 and ck.ex == 0:
            us = ks if side == "r" else 1 - ks
            l = cv2.perspectiveTransform(np.float32([[[us, 0], [us, 1]]]), Hm)[0]
            B.line(tuple(l[0]), tuple(l[1]), acc["hi"], 0.9, 3 * R)
        return None

    def f_hexwall(self, Fr, ck, tc, box, acc, _):
        inf = self.info
        if not inf:
            return
        side = self._side(tc)
        k = ck.p(8, 12, 0.7)
        tw = K.text_w("SERVANT  //  " + inf["en"], fnt("label_b", 24 * R), 4 * R) + 60 * R
        x0 = W - M - tw if side == "r" else M
        y0 = M * 0.8
        outline = K.notch_rect(x0, y0, x0 + tw, y0 + 56 * R, 14 * R, (1, 0, 1, 0))
        K.poly_grow(Fr, outline, k, acc["main"], 1.0, 1.8 * R)
        if k > 0.5:
            HE.fpoly(Fr, outline, PANEL, 0.6 * (k - 0.5) * 2)
            ttext(Fr, (x0 + 30 * R, y0 + 13 * R), "SERVANT  //  " + inf["en"], fnt("label_b", 24 * R), self.ink, INK, 4 * R, decode=ck.raw(12, 14), seed=ck.f)
        if inf.get("jp"):
            ttext(Fr, (x0 + 30 * R, y0 + 72 * R), inf["jp"], fnt("jp", 32 * R), self.ink, INK, 2 * R, rise=ck.raw(16, 14) * ck.p(0, 1, 0.8))

    # ============================================================ np
    def t_np(self, B, F, ck, tc, box, acc):
        cx, cy = tc
        rng = np.random.default_rng(self.seed)
        for j in range(18):
            a_ = rng.uniform(0, 2 * math.pi)
            r0 = 0.18 * H + rng.random() * 0.05 * H
            kk = ck.p(2 + (j % 6) * 0.8, 10, 0.2)
            L_ = (0.3 + rng.random() * 0.7) * W * kk
            if L_ > 1:
                B.line((cx + math.cos(a_) * r0, cy + math.sin(a_) * r0), (cx + math.cos(a_) * (r0 + L_), cy + math.sin(a_) * (r0 + L_)), self.ink,
                       0.35, (1.0 + 1.5 * (j % 3 == 0)) * R)
        return None

    def f_np(self, Fr, ck, tc, box, acc, _):
        inf = self.info
        if not inf or not (inf.get("np_jp") or inf.get("np_en")):
            return
        bw_, bh_ = int(W * 1.25), int(200 * R)
        band = Canvas.__new__(Canvas)
        band.txt = Image.new("RGBA", (bw_, bh_), (0, 0, 0, 0)); band.d = ImageDraw.Draw(band.txt)
        d = band.d
        kb = ck.p(4, 12, 0.3)
        d.rectangle([0, int(28 * R), bw_, bh_ - int(18 * R)], fill=PANEL + (int(170 * kb),))
        d.rectangle([0, int(20 * R), int(bw_ * kb), int(26 * R)], fill=tuple(acc["main"]) + (255,))
        d.rectangle([int(bw_ * (1 - kb)), bh_ - int(16 * R), bw_, bh_ - int(13 * R)], fill=self.ink + (150,))
        bx = int(W * 0.13 + M)
        kr = ck.p(8, 10, 0.5, ease=back_out)
        if kr > 0.02 and inf.get("np_rank"):               # 等级徽章
            s = cl(kr)
            d.rectangle([bx, int(40 * R), bx + int(130 * R * s), int(40 * R + 120 * R * s)], fill=tuple(acc["main"]) + (255,))
            if kr > 0.8:
                d.text((bx + 14 * R, 44 * R), "RANK", font=fnt("label_b", 15 * R), fill=ON_ACC + (255,))
                rk = inf["np_rank"]
                fr_ = fnt("display", 84 * R if len(rk) <= 3 else 56 * R)
                d.text((bx + 12 * R, 58 * R), rk, font=fr_, fill=ON_ACC + (255,))
        tx = bx + 160 * R
        name = inf.get("np_jp") or inf["np_en"]
        f_ = fnt("np", 92 * R) if inf.get("np_jp") else fnt("h1", 90 * R)
        if inf.get("np_ruby"):
            ttext(band, (tx, 30 * R), inf["np_ruby"], fnt("jp_m", 22 * R), self.ink, 0.85 * kb, 6 * R, rise=ck.raw(10, 14))
        ttext(band, (tx, 52 * R), name, f_, self.ink, 1.0 * kb, 2 * R, rise=ck.raw(10, 16))
        mx = tx + K.text_w(name, f_, 2 * R) + 44 * R
        if inf.get("np_jp") and inf.get("np_en"):
            ttext(band, (mx, 66 * R), inf["np_en"], fnt("h2", 32 * R), self.ink, INK * kb, 4 * R, decode=ck.raw(16, 12), seed=ck.f)
        if inf.get("np_type"):
            fl = fnt("label_b", 16 * R)
            tw = K.text_w(inf["np_type"], fl, 3 * R)
            kc = ck.p(20, 8, 0.6)
            d.rectangle([mx, 112 * R, mx + (tw + 20 * R) * kc, 140 * R], outline=self.ink + (220,), width=max(1, int(1.4 * R)))
            if kc > 0.9:
                ttext(band, (mx + 10 * R, 114 * R), inf["np_type"], fl, self.ink, INK, 3 * R)
            ttext(band, (mx + tw + 40 * R, 116 * R), inf["en"] + (("  /  " + inf["cls"]) if inf.get("cls") else ""), fnt("label", 15 * R),
                  acc["hi"], kc, 3 * R)
        im = band.txt.rotate(6, resample=Image.BICUBIC, expand=True)
        k = ck.p(0, 10, 0.3)
        cy = tc[1]
        by = (H * 0.74 if cy < H * 0.55 else H * 0.26) - im.height / 2
        bxo = -W * 0.12 - (1 - k) * W * 0.6
        Fr.txt.alpha_composite(im, (int(bxo), int(by)))

    # ============================================================ circle
    def _circle_img(self, acc):
        p = os.path.join(ROOT, "assets", "gen", "magic_circle.png")
        key = (p, tuple(acc["main"]))
        if key not in _CIRCLE and os.path.exists(p):
            im = Image.open(p).convert("RGBA")
            a = np.asarray(im).astype(np.float32)
            lum = a[..., :3].mean(2, keepdims=True) / 255.0
            col = np.array(acc["hi"], np.float32) * 0.55 + np.array((255, 236, 190), np.float32) * 0.45
            out = np.concatenate([np.clip(col * (0.55 + 0.6 * lum), 0, 255), a[..., 3:4]], 2).astype(np.uint8)
            _CIRCLE.clear(); _CIRCLE[key] = Image.fromarray(out, "RGBA")
        return _CIRCLE.get(key)

    def t_circle(self, B, F, ck, tc, box, acc):
        x0, y0, x1, y1 = box
        foot = ((x0 + x1) / 2, min(H * 0.93, y1 - 0.02 * H))
        rx = float(np.clip(1.2 * (x1 - x0), 0.32 * W, 0.48 * W)); sq = 0.28
        g = ck.p(0, 18, 0.2)
        im = self._circle_img(acc)
        if im is not None and g > 0.02:
            rot = (ck.f // 2) * 0.7
            r_ = im.rotate(rot, resample=Image.BILINEAR)
            r_ = r_.resize((max(2, int(2 * rx * g)), max(2, int(2 * rx * sq * g))), Image.BILINEAR)
            if g < 1:
                r_.putalpha(r_.getchannel("A").point(lambda v: int(v * g)))
            B.txt.alpha_composite(r_, (int(foot[0] - r_.width / 2), int(foot[1] - r_.height / 2)))
        elif g > 0.02:
            for s_, th in ((1.0, 2.4), (0.9, 1.2), (0.62, 1.6)):
                K.arc(B, foot, rx * s_ * g, 0, 360, self.ink, 0.85, th * R, sy=sq)
        ko = ck.p(10, 14, 0.4)
        K.arc(B, foot, rx * 1.06, (ck.f * 0.8) % 360, (ck.f * 0.8) % 360 + 70 * ko, acc["main"], 1.0, 4 * R, sy=sq)
        return None

    def f_circle(self, Fr, ck, tc, box, acc, _):
        inf = self.info
        if not inf:
            return
        k = ck.p(10, 16, 0.7)
        tr = 4 * R + 22 * R * (1 - k)
        s = ("SERVANT   " + inf["cls"] + "   " + inf["en"]) if inf.get("cls") else inf["en"]
        ttext(Fr, (W / 2, M * 0.9), s, fnt("label_b", 26 * R), self.ink, INK * k, tr, align="c", decode=ck.raw(10, 16), seed=ck.f)
        w_ = K.text_w(s, fnt("label_b", 26 * R), tr)
        K.grow_line(Fr, (W / 2 - w_ / 2, M * 0.9 + 44 * R), (W / 2 + w_ / 2, M * 0.9 + 44 * R), ck.p(12, 14, 0.6), acc["main"], 1.0, 2 * R, from_mid=True)
        if inf.get("jp"):
            ttext(Fr, (W / 2, M * 0.9 + 58 * R), inf["jp"], fnt("jp", 30 * R), self.ink, INK, 6 * R, align="c", rise=ck.raw(16, 16) * ck.p(0, 1, 0.8))

    # ============================================================ scope
    def t_scope(self, B, F, ck, tc, box, acc):
        cx, cy = tc
        r = float(np.clip(0.5 * max(box[2] - box[0], box[3] - box[1]), 0.20 * H, 0.32 * H))
        rot = self.rot0 + 0.15 * ck.f
        for j in range(4):                                  # 四段粗弧（间隙对着十字）
            kj = ck.p(2 + j * 1.5, 10, 0.3)
            a0 = rot + j * 90 + 8
            K.sector(B, (cx, cy), r, r + 7 * R, a0, a0 + 74 * kj, self.ink if j % 2 else acc["main"], 0.85)
        kt = ck.p(4, 12, 0.4)
        K.ticks(B, (cx, cy), r * 0.9, 60, 14 * R * kt, 5 * R * kt, -rot * 0.5, self.ink, 0.55 * kt, 1.2 * R, every=5)
        kc = ck.p(0, 12, 0.2)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # 间隙十字：内外两段 + 毫位刻度
            K.grow_line(B, (cx + dx * r * 0.22, cy + dy * r * 0.22), (cx + dx * r * 0.62, cy + dy * r * 0.62), kc, self.ink, 0.8, 1.6 * R)
            far = (W if dx > 0 else 0) if dx else (H if dy > 0 else 0)
            L_ = abs(far - (cx if dx else cy)) - r * 1.12
            if L_ > 0:
                p0 = (cx + dx * r * 1.12, cy + dy * r * 1.12)
                K.grow_line(B, p0, (p0[0] + dx * L_, p0[1] + dy * L_), kc, self.ink, 0.35, 1.2 * R)
            for m_ in range(1, 4):
                q = (cx + dx * r * (0.22 + 0.13 * m_), cy + dy * r * (0.22 + 0.13 * m_))
                if kc > 0.5:
                    B.line((q[0] - dy * 6 * R, q[1] - dx * 6 * R), (q[0] + dy * 6 * R, q[1] + dx * 6 * R), self.ink, 0.7, 1.4 * R)
        return r

    def f_scope(self, Fr, ck, tc, box, acc, r):
        inf = self.info
        side = self._side(tc)
        if inf:
            x = W - M if side == "r" else M
            al = "r" if side == "r" else "l"
            ttext(Fr, (x, H - M - 110 * R), inf["en"], fnt("h1", 56 * R), self.ink, INK * ck.p(10, 4, 0.7), 2 * R, align=al, decode=ck.raw(10, 14), seed=ck.f)
            if inf.get("jp"):
                ttext(Fr, (x, H - M - 44 * R), inf["jp"], fnt("jp", 30 * R), self.ink, INK, 2 * R, align=al, rise=ck.raw(14, 14) * ck.p(0, 1, 0.8))

    # ============================================================ link（v10 第二版：双人标注，替换 versus）
    # 用户看过 v10："HUD 的 XXX VS XXX 处理有些出戏，并且这种设计有点太幼稚" → 不要 VS / PLAYER 名牌 / 巨字。
    # 两个目标各四个角标（细 L 形，角色签名色）+ 外侧小号名字 / 职阶；两人之间一条虚发丝线（逐段画出）+ 刻度 + 中点切口，
    # 线上方写实测屏幕距离（跟踪框中心，px）。信息只有核对过的名字 / 职阶 + 实测量。
    def _vs(self):
        v = self.vs or {}
        from hud_v4 import INFO as _INFO
        keys = v.get("keys") or [None, None]
        infs = [_INFO.get(k) if k else None for k in keys]
        bx = v.get("boxes")
        if bx is None:
            bx = [[(p[0] - (p[2] if len(p) > 2 else 0.12) / 2) * W, (p[1] - (p[3] if len(p) > 3 else 0.20) / 2) * H,
                   (p[0] + (p[2] if len(p) > 2 else 0.12) / 2) * W, (p[1] + (p[3] if len(p) > 3 else 0.20) / 2) * H] for p in v.get("pts", [(0.35, 0.5), (0.65, 0.5)])]
        return keys, infs, [tuple(map(float, b)) for b in bx]

    def t_link(self, B, F, ck, tc, box, acc):
        keys, infs, bx = self._vs()
        ink = self.ink
        c = [((b[0] + b[2]) / 2, b[1] + (b[3] - b[1]) * 0.38) for b in bx]
        k = ck.p(8, 18, 0.4)                                # 虚发丝线：从两端向中间画
        if k > 0.02:
            mx, my = (c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2
            for p0 in c:
                q = (p0[0] + (mx - p0[0]) * k, p0[1] + (my - p0[1]) * k)
                _dash(B, p0, q, ink, 0.55, 1.2 * R, 14 * R, 8 * R)
            L_ = math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
            if L_ > 1 and k > 0.6:
                ux, uy = (c[1][0] - c[0][0]) / L_, (c[1][1] - c[0][1]) / L_
                nx, ny = -uy, ux
                for j in range(1, int(L_ / (40 * R))):      # 刻度
                    d = j * 40 * R
                    p = (c[0][0] + ux * d, c[0][1] + uy * d)
                    h = (7 if j % 5 == 0 else 3.5) * R
                    B.line((p[0] - nx * h, p[1] - ny * h), (p[0] + nx * h, p[1] + ny * h), ink, 0.45 * cl((k - 0.6) / 0.4), 1.0 * R)
        return None

    def f_link(self, Fr, ck, tc, box, acc, _):
        keys, infs, bx = self._vs()
        ink = self.ink
        cols = [tuple(int(v) for v in (SIG[k_][0] if k_ in SIG else acc["main"])) for k_ in keys]
        order = sorted((0, 1), key=lambda j: bx[j][0] + bx[j][2])
        for side, j in enumerate(order):                    # side 0 = 画面左的那个
            b = bx[j]
            kb = ck.p(2 + 4 * side, 12, 0.5 + 0.1 * side, ease=back_out)
            if kb <= 0.02:
                continue
            s_ = 1 + 0.35 * (1 - kb)
            m_ = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            x0, y0, x1, y1 = (m_[i % 2] + (b[i] - m_[i % 2]) * s_ for i in range(4))
            L_ = min(28 * R, (x1 - x0) * 0.3, (y1 - y0) * 0.3)
            for (px, py, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)):
                Fr.line((px, py), (px + sx * L_, py), cols[j], cl(kb), 2.0 * R)
                Fr.line((px, py), (px, py + sy * L_), cols[j], cl(kb), 2.0 * R)
            inf = infs[j]
            if not inf:
                continue
            kt = ck.p(8 + 4 * side, 12, 0.6 + 0.1 * side)
            al = "r" if side == 0 else "l"
            xt = x0 - 16 * R if side == 0 else x1 + 16 * R
            yt = float(np.clip(y0 + 6 * R, M, H - M - 90 * R))
            Fr.rect((xt - (3 * R if side == 0 else 0), yt + 4 * R), (xt + (0 if side == 0 else 3 * R), yt + 52 * R), cols[j], kt)
            xx = xt - 12 * R if side == 0 else xt + 12 * R
            en = inf["en"] if len(inf["en"]) <= 16 else short_name(inf) + (" [ALTER]" if "ALTER" in inf["en"] else "")
            ttext(Fr, (xx, yt), en, fnt("h2", 34 * R), ink, INK * kt, 3 * R, align=al, decode=ck.raw(8 + 4 * side, 14), seed=self.seed + j + ck.f)
            if inf.get("cls"):
                cw = ttext(Fr, (xx, yt + 40 * R), inf["cls"], fnt("label", 15 * R), ink, 0.7 * kt, 3 * R, align=al)
                if inf.get("jp"):
                    ttext(Fr, (xx + cw + 14 * R if side else xx - cw - 14 * R, yt + 37 * R), inf["jp"], fnt("jp_m", 17 * R), cols[j], 0.9 * kt, 2 * R, align=al,
                          rise=ck.raw(12 + 4 * side, 14))
        kd = ck.p(18, 12, 0.8)                              # 中点：切口 + 实测距离
        if kd > 0.02:
            c = [((b[0] + b[2]) / 2, b[1] + (b[3] - b[1]) * 0.38) for b in bx]
            mx, my = (c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2
            dist = math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
            s_ = 7 * R
            Fr.tri([(mx, my - s_), (mx + s_, my), (mx, my + s_)], ink, 0.9 * kd)
            Fr.tri([(mx, my - s_), (mx - s_, my), (mx, my + s_)], acc["main"], 0.9 * kd)
            ttext(Fr, (mx, my - 34 * R), "%04d px" % int(round(dist * expo_out(ck.raw(18, 16)))), fnt("micro", 15 * R), ink, 0.75 * kd, 3 * R, align="c")

    # ============================================================ blueprint（v10 第二版：技术制图标注）
    # 目标框当作被测零件：延伸线 + 带箭头的尺寸线（实测框宽 / 高，px，数字从 0 走到实测值）+ 点划中心线 + 间隙中心标；
    # 三条引线编号标注（01 真名 / 02 职阶 / 03 宝具等级·种类）；远侧下角图框标题栏（细线表格逐格画出、逐行解码）。
    def _bp_box(self, box):
        x0, y0, x1, y1 = box
        x0, x1 = max(M * 0.6, x0), min(W - M * 0.6, x1)
        y0, y1 = max(M * 0.6, y0), min(H - M * 0.6, y1)
        return x0, y0, x1, y1

    def t_blueprint(self, B, F, ck, tc, box, acc):
        x0, y0, x1, y1 = self._bp_box(box)
        ink = self.ink
        side = self._side(tc)
        k = ck.p(0, 16, 0.3)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for p, q in (((x0 - 60 * R, cy), (x1 + 60 * R, cy)), ((cx, y0 - 60 * R), (cx, y1 + 60 * R))):   # 点划中心线
            q2 = (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k)
            _dash(B, p, q2, ink, 0.30, 1.0 * R, 26 * R, 6 * R, dot=True)
        g, L_ = 10 * R, 26 * R * cl(k)                      # 间隙中心标
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            B.line((cx + dx * g, cy + dy * g), (cx + dx * (g + L_), cy + dy * (g + L_)), acc["main"], 0.9, 1.8 * R)
        yd = min(H - M * 0.55, y1 + 40 * R)                 # 横向尺寸线（框下方）
        kx = ck.p(4, 14, 0.4)
        if kx > 0.02:
            for x in (x0, x1):
                K.grow_line(B, (x, y1 + 6 * R), (x, yd + 14 * R), kx, ink, 0.5, 1.0 * R)
            _dim(B, (x0, yd), (x1, yd), kx, ink, 0.85)
        xd = (x1 + 40 * R) if side == "r" else (x0 - 40 * R)   # 纵向尺寸线（远侧）
        xd = float(np.clip(xd, M * 0.5, W - M * 0.5))
        ky = ck.p(7, 14, 0.5)
        if ky > 0.02:
            for y in (y0, y1):
                K.grow_line(B, ((x1 + 6 * R) if side == "r" else (x0 - 6 * R), y), (xd + (14 * R if side == "r" else -14 * R), y), ky, ink, 0.5, 1.0 * R)
            _dim(B, (xd, y0), (xd, y1), ky, ink, 0.85)
        return (x0, y0, x1, y1, yd, xd)

    def f_blueprint(self, Fr, ck, tc, box, acc, geo):
        x0, y0, x1, y1, yd, xd = geo
        ink = self.ink
        inf = self.info or {}
        side = self._side(tc)
        fm = fnt("micro", 16 * R)
        kx, ky = ck.p(8, 12, 0.45), ck.p(11, 12, 0.55)
        if kx > 0.05:                                       # 尺寸数字（实测，数到位）
            ttext(Fr, ((x0 + x1) / 2, yd - 28 * R), "W  %04d" % int(round((x1 - x0) * expo_out(ck.raw(8, 18)))), fm, ink, 0.9 * kx, 3 * R, align="c")
        if ky > 0.05:
            ttext(Fr, (xd + (14 * R if side == "r" else -14 * R), (y0 + y1) / 2 - 10 * R), "H  %04d" % int(round((y1 - y0) * expo_out(ck.raw(11, 18)))),
                  fm, ink, 0.9 * ky, 3 * R, align="l" if side == "r" else "r")
        rows = []                                           # 引线编号标注（只写核对过的）
        if inf.get("en"):
            rows.append(("01", inf["en"], (( x0 + x1) / 2, y0 + (y1 - y0) * 0.12)))
        if inf.get("cls"):
            rows.append(("02", "CLASS  " + inf["cls"], ((x0 + x1) / 2, (y0 + y1) / 2)))
        if inf.get("np_rank") and inf["np_rank"] not in ("—", ""):
            rows.append(("03", "N.P.  " + inf["np_rank"] + (("  /  " + inf["np_type"]) if inf.get("np_type") else ""), ((x0 + x1) / 2, y0 + (y1 - y0) * 0.75)))
        xl = float(np.clip(x1 + 170 * R, M, W - M - 420 * R)) if side == "r" else float(np.clip(x0 - 170 * R, M + 420 * R, W - M))
        ybase = float(np.clip(y0 + 10 * R, M + 30 * R, H - M - 330 * R))
        for i, (num, txt, anc) in enumerate(rows):
            kl = ck.p(12 + 4 * i, 12, 0.6 + 0.08 * i)
            if kl <= 0.02:
                continue
            yl = ybase + i * 70 * R
            elb = (xl - 34 * R, yl + 13 * R) if side == "r" else (xl + 34 * R, yl + 13 * R)
            Fr.rect((anc[0] - 4 * R, anc[1] - 4 * R), (anc[0] + 4 * R, anc[1] + 4 * R), acc["main"], cl(kl * 2))
            K.poly_grow(Fr, [anc, elb, (xl, yl + 13 * R)], kl, ink, 0.6, 1.2 * R, closed=False)
            if kl > 0.6:
                tx = xl if side == "r" else xl - 400 * R
                Fr.rect((tx, yl), (tx + 40 * R, yl + 26 * R), acc["main"], 1.0)
                ttext(Fr, (tx + 7 * R, yl + 2 * R), num, fnt("label_b", 17 * R), ON_ACC, 1.0, 2 * R)
                ttext(Fr, (tx + 54 * R, yl + 1 * R), txt, fnt("h3", 24 * R), ink, INK * cl((kl - 0.6) / 0.4), 2 * R, decode=ck.raw(14 + 4 * i, 12), seed=self.seed + i + ck.f)
        # 图框标题栏
        tw, rh = 470 * R, 30 * R
        lines = [("TITLE", inf.get("en", "SUBJECT")), ("読み", inf.get("jp", "")), ("CLASS", inf.get("cls", "")), ("WORK", self.work or "")]
        lines = [l for l in lines if l[1]]
        th_ = (len(lines) + 1) * rh
        bx0 = (W - M * 0.6 - tw) if side == "r" else (M * 0.6)
        by0 = H - M * 0.6 - th_
        kt = ck.p(16, 14, 0.75)
        if kt > 0.02:
            K.poly_grow(Fr, [(bx0, by0), (bx0 + tw, by0), (bx0 + tw, by0 + th_), (bx0, by0 + th_)], kt, ink, 0.85, 1.4 * R)
            if kt > 0.4:
                HE.fpoly(Fr, [(bx0, by0), (bx0 + tw, by0), (bx0 + tw, by0 + th_), (bx0, by0 + th_)], PANEL, 0.55 * cl((kt - 0.4) / 0.6))
            K.stripes(Fr, bx0 + 1, by0 + 1, bx0 + 110 * R * kt, by0 + rh - 1, acc["main"], 0.55)
            ttext(Fr, (bx0 + 124 * R, by0 + 6 * R), "SPEC SHEET", fnt("label_b", 16 * R), ink, kt, 5 * R)
            ttext(Fr, (bx0 + tw - 12 * R, by0 + 7 * R), "SCALE 1:1", fnt("micro", 14 * R), ink, 0.6 * kt, 2 * R, align="r")
            for i, (lab, val) in enumerate(lines):
                kr = ck.p(20 + 3 * i, 10, 0.8 + 0.03 * i)
                y = by0 + (i + 1) * rh
                Fr.line((bx0, y), (bx0 + tw * kr, y), ink, 0.45, 1.0 * R)
                Fr.line((bx0 + 110 * R, y), (bx0 + 110 * R, y + rh * kr), ink, 0.45, 1.0 * R)
                if kr > 0.3:
                    ttext(Fr, (bx0 + 12 * R, y + 7 * R), lab, fnt("micro", 14 * R) if ord(lab[0]) < 0x2E80 else fnt("jp_m", 15 * R), ink, 0.6 * kr, 3 * R)
                    f_ = fnt("jp_m", 18 * R) if any(ord(ch) > 0x2E80 for ch in val) else fnt("h3", 20 * R)
                    ttext(Fr, (bx0 + 124 * R, y + 4 * R), val, f_, ink, INK * kr, 2 * R, decode=ck.raw(20 + 3 * i, 12), seed=self.seed + 7 * i + ck.f)

    # ============================================================ codex（v10 第二版：竖排编辑式）
    # 远侧一列大号明朝竖排日文真名（世界层，被人物遮挡，逐字自上而下显影）+ 一道角色色双细线从上往下长；
    # 竖排底部一块小号版式：英文名（字距收拢）/ 职阶 / 宝具日文名 + 读音（核对过的才写）；一条发丝线从版式块引到目标。
    def t_codex(self, B, F, ck, tc, box, acc):
        inf = self.info or {}
        name = re.sub(r"[〔〕\[\]（）]", "", inf.get("jp", "")) if inf else ""
        if not name:
            return None
        side = self._side(tc)
        n = len(name)
        fs = min(0.118 * H, 0.70 * H / max(1, n))
        f_ = fnt("np", fs)
        x = (W - M * 1.1 - fs) if side == "r" else (M * 1.1)
        y0 = M * 0.9
        k = ck.raw(2, 10 + 2.2 * n)
        step = fs * 1.04
        for i, ch in enumerate(name):                        # 竖排真名：逐字自上而下显影
            kk = cl(k * (n + 2) - i)
            if kk <= 0:
                break
            B.d.text((x, y0 + i * step - (1 - expo_out(kk)) * 14 * R), ch, font=f_, fill=tuple(self.ink) + (int(255 * INK * expo_out(kk)),))
        kr = ck.p(0, 18, 0.2)                               # 角色色双细线
        xr = x - 22 * R if side == "r" else x + fs + 22 * R
        B.line((xr, y0), (xr, y0 + (n * step) * kr), acc["main"], 0.95, 2.4 * R)
        xr2 = xr + (-7 * R if side == "r" else 7 * R)
        B.line((xr2, y0), (xr2, y0 + (n * step) * kr * 0.82), self.ink, 0.5, 1.0 * R)
        # 版式块：英文名（字距收拢）/ 职阶 / 宝具读音 + 日文名 + 英文名·等级
        al = "r" if side == "r" else "l"
        xa = xr2 - 16 * R if side == "r" else xr2 + 16 * R
        ya = float(np.clip(y0 + n * step - 230 * R, H * 0.40, H - M - 240 * R))
        k = ck.p(10, 14, 0.6)
        if k > 0.02:
            tr = (3 + 14 * (1 - expo_out(ck.raw(10, 20)))) * R
            ttext(B, (xa, ya), inf.get("en", ""), fnt("h2", 38 * R), self.ink, INK * k, tr, align=al)
            y = ya + 52 * R
            B.line((xa - (220 * R if side == "r" else 0), y), (xa + (0 if side == "r" else 220 * R), y), acc["main"], k, 1.6 * R)
            if inf.get("cls"):
                ttext(B, (xa, y + 12 * R), "SERVANT  ·  " + inf["cls"], fnt("label", 17 * R), self.ink, 0.75 * ck.p(13, 10, 0.65), 5 * R, align=al)
            if inf.get("np_jp"):
                kn = ck.p(16, 12, 0.7)
                if inf.get("np_ruby"):
                    ttext(B, (xa, y + 48 * R), inf["np_ruby"], fnt("jp_m", 15 * R), self.ink, 0.7 * kn, 4 * R, align=al, rise=ck.raw(16, 14))
                ttext(B, (xa, y + 70 * R), inf["np_jp"], fnt("np", 40 * R), self.ink, INK * kn, 2 * R, align=al, rise=ck.raw(17, 16))
                if inf.get("np_en"):
                    ttext(B, (xa, y + 122 * R), inf["np_en"] + (("   RANK " + inf["np_rank"]) if inf.get("np_rank") else ""), fnt("label", 15 * R),
                          acc["hi"], 0.85 * kn, 3 * R, align=al, decode=ck.raw(18, 14), seed=ck.f)
        kl = ck.p(20, 14, 0.85)                             # 发丝引线 → 目标
        if kl > 0.02:
            p0 = (xa + (-240 * R if side == "r" else 240 * R), ya + 26 * R)
            K.grow_line(B, p0, tc, kl, self.ink, 0.4, 1.0 * R)
            B.rect((tc[0] - 4 * R, tc[1] - 4 * R), (tc[0] + 4 * R, tc[1] + 4 * R), acc["main"], cl(kl * 1.5))
        return None

    def f_codex(self, Fr, ck, tc, box, acc, geo):
        return None

    # ============================================================ track（v10 新增：运动轨迹）
    # 目标框中心的真实跟踪轨迹（引擎传入本镜开头到当前帧的路径）：尾迹折线（近段强调色）+ 每 6 帧一个刻度、每 0.5 s 一个时间标；
    # 当前点间隙十字（不画圆）+ 航向箭头；远侧读数块：TRACK // 名字、屏幕速度 px/s、航向角、锁定时长 —— 全部由轨迹算出。
    def _trk(self):
        p = np.asarray(self.path if self.path is not None else np.zeros((0, 2)), np.float64)
        if len(p) < 2:
            return p, 0.0, 0.0
        q = p[-7:]
        v = (q[-1] - q[0]) / max(1, len(q) - 1) * 60.0
        return p, float(np.hypot(*v)), float((math.degrees(math.atan2(v[0], -v[1])) + 360) % 360)

    def t_track(self, B, F, ck, tc, box, acc):
        p, spd, hdg = self._trk()
        if len(p) < 2:
            return None
        n = len(p)
        lo = max(0, n - 90)
        kt = ck.p(0, 10, 0.2)
        for j in range(lo + 1, n):
            a_ = (j - lo) / max(1, n - lo)
            B.line(tuple(p[j - 1]), tuple(p[j]), acc["main"] if j > n - 20 else self.ink, (0.25 + 0.65 * a_) * kt, (1.2 + 1.6 * a_) * R)
        for j in range(n - 1, lo - 1, -6):
            q = p[j]; o = n - 1 - j
            s_ = (7 if o % 30 == 0 else 4) * R
            B.line((q[0] - s_, q[1]), (q[0] + s_, q[1]), self.ink, 0.6 * kt, 1.2 * R)
            if o % 30 == 0 and o > 0:
                ttext(B, (q[0] + 10 * R, q[1] - 22 * R), "-%.1fs" % (o / 60.0), fnt("micro", 13 * R), self.ink, 0.55 * kt, 1 * R)
        return None

    def f_track(self, Fr, ck, tc, box, acc, _):
        p, spd, hdg = self._trk()
        inf = self.info or {}
        cx, cy = (p[-1] if len(p) else tc)
        kc = ck.p(2, 10, 0.4, ease=back_out)                # 间隙十字 + 航向箭头
        if kc > 0.02:
            g, L_ = 16 * R, 54 * R * cl(kc)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                Fr.line((cx + dx * g, cy + dy * g), (cx + dx * (g + L_), cy + dy * (g + L_)), acc["main"], 0.95, 2.4 * R)
            if spd > 30:
                u = math.radians(hdg)
                d_ = (math.sin(u), -math.cos(u))
                tip = (cx + d_[0] * (g + L_ + 46 * R), cy + d_[1] * (g + L_ + 46 * R))
                base = (cx + d_[0] * (g + L_ + 12 * R), cy + d_[1] * (g + L_ + 12 * R))
                nx, ny = -d_[1], d_[0]
                Fr.tri([tip, (base[0] + nx * 12 * R, base[1] + ny * 12 * R), (base[0] - nx * 12 * R, base[1] - ny * 12 * R)], self.ink, 0.9 * cl(kc))
        side = self._side((cx, cy))
        x = W - M - 420 * R if side == "r" else M
        y = float(np.clip(cy - 150 * R, M + 60 * R, H - M - 330 * R))
        kp = ck.p(6, 12, 0.6)
        if kp <= 0.02:
            return
        outline = K.notch_rect(x, y, x + 420 * R, y + 196 * R, 14 * R, (1, 0, 0, 1))
        K.poly_grow(Fr, outline, kp, acc["main"], 1.0, 1.6 * R)
        if kp > 0.5:
            HE.fpoly(Fr, outline, PANEL, 0.6 * (kp - 0.5) * 2)
        Fr.rect((x, y), (x + 132 * R * kp, y + 26 * R), acc["main"], 1.0)
        ttext(Fr, (x + 10 * R, y + 3 * R), "TRACK", fnt("label_b", 16 * R), ON_ACC, kp, 4 * R)
        ttext(Fr, (x + 150 * R, y + 4 * R), inf.get("en", "OBJECT"), fnt("micro", 15 * R), self.ink, INK * kp, 2 * R, decode=ck.raw(8, 12), seed=ck.f)
        fm, fb = fnt("micro", 14 * R), fnt("h1", 40 * R)
        rows = (("SPD", "%5.0f" % spd, "px/s"), ("HDG", "%03.0f" % hdg, "deg"), ("T+", "%05.2f" % (len(p) / 60.0), "s"))
        for j, (lab, val, unit) in enumerate(rows):
            kr = ck.p(10 + 3 * j, 8, 0.8)
            yy = y + 40 * R + j * 50 * R
            ttext(Fr, (x + 22 * R, yy + 14 * R), lab, fm, self.ink, 0.6 * kr, 3 * R)
            ttext(Fr, (x + 100 * R, yy), val, fb, self.ink, INK * kr, 2 * R)
            ttext(Fr, (x + 250 * R, yy + 14 * R), unit, fm, self.ink, 0.5 * kr, 2 * R)
        if len(p) > 7:                                      # 速度条：相对本镜到目前为止的最大速度
            vs_ = [float(np.hypot(*((p[j] - p[j - 6]) / 6 * 60))) for j in range(6, len(p))]
            K.seg_meter(Fr, x + 300 * R, y + 52 * R, 100 * R, 12 * R, 10, int(round(10 * spd / max(1.0, max(vs_)))), acc["main"], kp)

    # ============================================================ viewfinder（v10 新增：摄影机取景器）
    # 半透明 2.39:1 遮幅 + 遮幅内四角标 + 三分线发丝 + 中心小十字；左上 REC + 成片时间码（时:分:秒:帧，真实）；右上作品名；
    # 对焦括号跟着目标（AF // 名字）；下遮幅里一条按真实音乐电平走的电平表。不写任何编造的镜头参数（光圈 / ISO 之类）。
    def t_viewfinder(self, B, F, ck, tc, box, acc):
        return None

    def f_viewfinder(self, Fr, ck, tc, box, acc, _):
        k = ck.p(0, 12, 0.1)
        if getattr(self, "cine", False):                    # 外面已经有 21:9 遮幅（v7_cine）：不再画自己的遮幅，取景框贴着遮幅内沿
            hb = (H - W * 9.0 / 21.0) / 2
        else:
            hb = (H - W / 2.39) / 2 * k
            Fr.rect((0, 0), (W, hb), (6, 6, 8), 0.55)
            Fr.rect((0, H - hb), (W, H), (6, 6, 8), 0.55)
        L_ = 70 * R
        y0, y1, x0, x1 = hb + 26 * R, H - hb - 26 * R, M * 0.8, W - M * 0.8
        for sx, sy, x, y in ((1, 1, x0, y0), (-1, 1, x1, y0), (-1, -1, x1, y1), (1, -1, x0, y1)):
            HE.poly(Fr, [(x + sx * L_ * k, y), (x, y), (x, y + sy * L_ * k)], self.ink, 0.85, 2.0 * R, closed=False)
        kg = ck.p(4, 14, 0.3)
        for q in (1, 2):
            Fr.line((W * q / 3, y0 + 30 * R), (W * q / 3, y0 + 30 * R + (y1 - y0 - 60 * R) * kg), self.ink, 0.18, 1.0 * R)
            Fr.line((x0 + 30 * R, H * q / 3), (x0 + 30 * R + (x1 - x0 - 60 * R) * kg, H * q / 3), self.ink, 0.18, 1.0 * R)
        Fr.line((W / 2 - 18 * R, H / 2), (W / 2 + 18 * R, H / 2), self.ink, 0.6 * k, 1.4 * R)
        Fr.line((W / 2, H / 2 - 18 * R), (W / 2, H / 2 + 18 * R), self.ink, 0.6 * k, 1.4 * R)
        Fi = int(round(self.t_now * 60))
        fm, fl = fnt("micro", 16 * R), fnt("label_b", 20 * R)
        yt = (hb + 14 * R) if getattr(self, "cine", False) else max(14 * R, hb * 0.5 - 12 * R)
        if (ck.f // 30) % 2 == 0:
            Fr.rect((x0, yt + 4 * R), (x0 + 14 * R, yt + 18 * R), (226, 40, 40), k)
        ttext(Fr, (x0 + 26 * R, yt), "REC", fl, self.ink, k, 4 * R)
        ttext(Fr, (x0 + 100 * R, yt + 2 * R), "TC %02d:%02d:%02d:%02d" % (Fi // 216000, Fi // 3600 % 60, Fi // 60 % 60, Fi % 60), fm, self.ink, 0.85 * k, 3 * R)
        ttext(Fr, (x0 + 380 * R, yt + 2 * R), "60P   2.39 : 1", fm, self.ink, 0.55 * k, 3 * R)
        if self.work:
            ttext(Fr, (x1, yt + 2 * R), self.work, fm, self.ink, 0.85 * k, 3 * R, align="r", decode=ck.raw(4, 14), seed=ck.f)
        fb = self._facebox(box, tc)                         # 对焦括号
        ka = ck.p(6, 10, 0.5, ease=back_out)
        if ka > 0.02:
            s_ = 1 + 0.5 * (1 - ka)
            m_ = ((fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2)
            bb = tuple(m_[i % 2] + (fb[i] - m_[i % 2]) * s_ for i in range(4))
            K.bracket(Fr, bb, 26 * R, 2.4 * R, acc["main"] if ck.f > 22 else self.ink, cl(ka))
            inf = self.info or {}
            ttext(Fr, (bb[0], bb[3] + 8 * R), "AF  //  " + (inf.get("en") or "SUBJECT"), fm, self.ink, 0.8 * cl(ka), 3 * R, decode=ck.raw(8, 12), seed=ck.f)
        if self.rms_seg is not None and self.rms_seg.size:  # 下遮幅：音乐电平（真实数据）
            yb = (H - hb - 36 * R) if getattr(self, "cine", False) else H - max(30 * R, hb * 0.5)
            n_ = len(self.rms_seg)
            for j, v in enumerate(self.rms_seg[::-1]):
                xx = W / 2 + (j - n_ / 2) * 9 * R
                hh = max(1.0, 18 * R * min(1.0, v / self.rms_top)) * k
                Fr.rect((xx, yb - hh), (xx + 5 * R, yb + hh), acc["main"] if j == 0 else self.ink, 0.9 if j == 0 else 0.4)
            ttext(Fr, (W / 2 - n_ / 2 * 9 * R - 70 * R, yb - 9 * R), "AUDIO", fm, self.ink, 0.5 * k, 3 * R)

    # ============================================================ 公共精密层（用户："HUD 可以做的精密复杂一些…信息密集…但是不能太乱"）
    def chrome(self, Fr, ck, tc, acc, t, rms, F=0):
        """全部贴网格：四角切口括号 + 角标（作品 / 曲时间 / 职阶·宝具种类 / 音乐电平）+ 左 / 上边轨（游标跟着目标）。微元素一律低透明度。"""
        e = 0.42 * M
        inf = self.info or {}
        k = ck.p(0, 10, 0.02)
        ink = self.ink
        L_ = 64 * R
        for sx, sy, x, y in ((1, 1, e, e), (-1, 1, W - e, e), (-1, -1, W - e, H - e), (1, -1, e, H - e)):
            n = 9 * R
            HE.poly(Fr, [(x + sx * L_ * k, y), (x + sx * n, y), (x, y + sy * n), (x, y + sy * L_ * k)], ink, 0.62, 1.4 * R, closed=False)
        fm = fnt("micro", 14 * R)
        if self.work:
            ttext(Fr, (e + 20 * R, e + 10 * R), self.work, fm, ink, 0.55 * k, 3 * R, decode=ck.raw(2, 12), seed=F)
        ttext(Fr, (W - e - 20 * R, e + 10 * R), "PROVANT  %02d:%05.2f" % (int(t // 60), t % 60), fm, ink, 0.55 * k, 2 * R, align="r")
        if inf.get("cls"):
            s_ = "CLASS %s" % inf["cls"] + ("   ·   %s" % inf["np_type"] if inf.get("np_type") else "")
            ttext(Fr, (e + 20 * R, H - e - 26 * R), s_, fm, ink, 0.55 * k, 3 * R, decode=ck.raw(4, 12), seed=F + 1)
        if rms is not None and len(rms):                    # 右下：最近 32 帧的音乐电平条（真实数据）
            seg = np.asarray(rms[max(0, F - 31):F + 1], np.float32)
            if seg.size:
                top = float(np.percentile(rms, 99)) or 1.0
                for j, v in enumerate(seg[::-1]):
                    x = W - e - 20 * R - j * 6 * R
                    hgt = max(1.0, 26 * R * min(1.0, v / top)) * k
                    Fr.rect((x - 3.5 * R, H - e - 12 * R - hgt), (x, H - e - 12 * R), acc["main"] if j == 0 else ink, 0.85 if j == 0 else 0.4)
        y0r, y1r = 0.22 * H, 0.78 * H                       # 左边轨 + 游标（目标纵坐标）
        kr = ck.p(2, 14, 0.04)
        K.grow_line(Fr, (e, y0r), (e, y1r), kr, ink, 0.35, 1.2 * R, from_mid=True)
        for j, y in enumerate(np.arange(y0r, y1r + 1, 24 * R)):
            if abs(y - (y0r + y1r) / 2) <= (y1r - y0r) / 2 * kr:
                Fr.line((e, y), (e + (12 if j % 5 == 0 else 5) * R, y), ink, 0.35, 1.0 * R)
        x0r, x1r = 0.30 * W, 0.70 * W                       # 上边轨 + 游标（目标横坐标）
        K.grow_line(Fr, (x0r, e), (x1r, e), kr, ink, 0.35, 1.2 * R, from_mid=True)
        for j, x in enumerate(np.arange(x0r, x1r + 1, 24 * R)):
            if abs(x - (x0r + x1r) / 2) <= (x1r - x0r) / 2 * kr:
                Fr.line((x, e), (x, e + (12 if j % 5 == 0 else 5) * R), ink, 0.35, 1.0 * R)
        km = ck.p(8, 10, 0.1)
        if km > 0.02:
            ty_ = float(np.clip(tc[1], y0r, y1r)); tx_ = float(np.clip(tc[0], x0r, x1r))
            s = 9 * R
            Fr.tri([(e + 16 * R, ty_), (e + 16 * R + s * 1.4, ty_ - s), (e + 16 * R + s * 1.4, ty_ + s)], acc["main"], km)
            Fr.tri([(tx_, e + 16 * R), (tx_ - s, e + 16 * R + s * 1.4), (tx_ + s, e + 16 * R + s * 1.4)], acc["main"], km)
            ttext(Fr, (e + 44 * R, ty_ - 9 * R), "TGT", fnt("micro", 13 * R), ink, 0.6 * km, 2 * R)

    # ============================================================ 放大窗（scope）
    def zoom(self, base, src, ck, tc, side, acc):
        k = ck.p(10, 10, 0.5)
        if src is None or k <= 0.02:
            return base
        cx, cy = tc
        sw, sh_ = 250 * R, 141 * R
        sx0 = int(np.clip(cx - sw / 2, 0, W - sw)); sy0 = int(np.clip(cy - sh_ / 2, 0, H - sh_))
        ww, wh = int(W * 0.28), int(W * 0.28 * 9 / 16)
        wx0 = int(W - M - ww) if side == "r" else int(M)
        wy0 = int(M + 40 * R)
        hh = max(1, int(wh * k))
        crop = cv2.resize(src[sy0:sy0 + int(sh_), sx0:sx0 + int(sw)], (ww, wh), interpolation=cv2.INTER_CUBIC)[(wh - hh) // 2:(wh - hh) // 2 + hh]
        out = base.copy()
        out[wy0 + (wh - hh) // 2:wy0 + (wh - hh) // 2 + hh, wx0:wx0 + ww] = crop
        cv = Canvas()
        HE.poly(cv, K.notch_rect(wx0, wy0, wx0 + ww, wy0 + wh, 18 * R, (1, 0, 1, 0)), self.ink, 0.9 * k, 1.6 * R)
        cv.rect((wx0 + 18 * R, wy0 - 30 * R), (wx0 + 18 * R + 150 * R * k, wy0), acc["main"], 1.0)
        if k > 0.9:
            ttext(cv, (wx0 + 30 * R, wy0 - 27 * R), "ZOOM  x3", fnt("label_b", 16 * R), ON_ACC, 1.0, 3 * R)
        K.bracket(cv, (sx0, sy0, sx0 + sw, sy0 + sh_), 18 * R, 2 * R, acc["main"], k)
        p0 = (sx0 + sw, sy0) if side == "r" else (sx0, sy0)
        p1 = (wx0, wy0 + wh) if side == "r" else (wx0 + ww, wy0 + wh)
        K.grow_line(cv, p0, p1, k, self.ink, 0.6, 1.2 * R)
        return cv.over(out)

    # ============================================================ 合成
    def render3(self, base, f, t, bk, bf, box, acc, rms, src=None, life_in=1.0, life_out=0.0, cam=None, cam_vel=(0.0, 0.0),
                fg=None, n_samples=1, burst=False, hud_glitch=0.0, gseed=0, kind="face", path=None, vs=None):
        if life_out >= 1.0:
            return base
        self.path, self.vs, self.t_now = path, vs, t
        Fi = int(round(t * 60))
        if rms is not None and len(rms):
            self.rms_top = float(np.percentile(rms, 99)) or 1.0
            self.rms_seg = np.asarray(rms[max(0, Fi - 63):Fi + 1], np.float32)
            self.rms_now = float(min(1.0, rms[min(Fi, len(rms) - 1)] / self.rms_top))
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        tc = ((x0 + x1) / 2, y0 + (0.40 if (bw > 0.30 * W or bh > 0.5 * H) and kind != "person" else 0.18 if kind == "person" else 0.40) * bh)
        tc = (float(np.clip(tc[0], 0.12 * W, 0.88 * W)), float(np.clip(tc[1], 0.14 * H, 0.86 * H)))
        ck = Clock(f, life_out)
        acc = make_pal(acc, (self.info or {}).get("key"))
        self.ink = acc["ink"]
        back, front = Canvas(), Canvas()
        r = getattr(self, "t_" + self.kind)(back, front, ck, tc, box, acc)
        getattr(self, "f_" + self.kind)(front, ck, tc, box, acc, r)
        self.chrome(front, ck, tc, acc, t, rms, F=int(round(t * 60)))
        out = base
        if self.kind == "scope" and n_samples:
            out = self.zoom(base, src, ck, tc, self._side(tc), acc)
        import gpu
        ax, ay = tc
        Mx = np.eye(3, dtype=np.float64)
        if cam is not None:
            C = np.eye(3); C[:2] = np.asarray(cam, np.float64)
            Mx = (np.eye(3) + self.depth * (C - np.eye(3))) @ Mx
        g = max(hud_glitch, 0.45 * life_out)
        if gpu.use():
            rows = np.zeros(H, np.float32)
            if g > 0.02:
                rng = np.random.default_rng(gseed)
                y = 0
                while y < H:
                    h = int(rng.integers(int(4 * R), int(50 * R))); u = rng.random()
                    if u < 0.5 * g:
                        rows[y:y + h] = rng.normal(0, 70 * R * g)
                    elif u < 0.58 * g:
                        rows[y:y + h] = 1e5
                    y += h
            fgu = None if fg is None else (np.clip(fg, 0, 1) * 255).astype(np.uint8)
            return gpu.hud3(out, back.a, np.asarray(back.txt), front.a, np.asarray(front.txt), np.linalg.inv(Mx), (ax, ay), 1e6, 80 * R,
                            1.0, 1.0, fg=fgu, rows=rows, vel=(cam_vel[0] * 0.25, cam_vel[1] * 0.25), rgbsplit=8 * R * g if g > 0.02 else 0.0)   # 文字面板要读得清：运动模糊只留 1/4
        Pb, Pf = premult(back), premult(front)
        if cam is not None:
            Pb = cv2.warpAffine(Pb, Mx[:2].astype(np.float32), (W, H))
        if fg is not None:
            Pb *= (1 - fg)[..., None]
        P = Pf + Pb * (1 - Pf[..., 3:4])
        if g > 0.02:
            P = glitch_rows(P, g, gseed)
        o = out.astype(np.float32) / 255 * (1 - P[..., 3:4]) + P[..., :3]
        return (np.clip(o, 0, 1) * 255).astype(np.uint8)

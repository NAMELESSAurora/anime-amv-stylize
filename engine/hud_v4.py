# -*- coding: utf-8 -*-
"""
HUD v4（用户 10-19 看过 v3 的第 1 / 2 / 4 条）：
  "进一步减少 HUD 的单帧画面密度，但是增加覆盖面积，减少 HUD 风格化的无用信息，信息要清楚且有意义，
   然后 HUD 的画面覆盖面积要大一些且要风格多变，现在的 HUD 还是太过于公式化（经常是把所有元素都堆砌在画面的一小部分）"
  "蜂巢阵可以作为转场，或者在 HUD 界面要覆盖大面积进行处理"  "可以做一些 FATE 系列的风格 HUD"

做法：每镜只用一种"模板"，每个模板只有 2–4 个大元素、铺满画面一侧或整幅；文字只写真实信息
（真名 / 日文名 / 职阶 / 宝具名与读音 / 等级与种类 / 参数——data/servant_info.json，FGO Wiki + Atlas Academy 核对过；
 查不到的角色只写名字和职阶，不编数字）。模板：
  scope    全幅十字准线 + 大圆 + 上 / 左边缘标尺（目标坐标三角标）+ 一个名字块 + 大采样放大窗
  frame    画幅四角大括号 + 底部全宽数据条（职阶 | 真名 | 宝具），左上作品名；目标只有一个小菱形
  hexwall  EVA 式大蜂巢铺满画面远离人物的那半边（波形逐格弹出），少数实心格写职阶 / 宝具等级，顶部真名条
  status   Fate 从者参数表（STR / END / AGI / MAN / LUK / NP 字母等级 + 槽），大面板占画面三分之一
  np       宝具名斜向横幅（日文宝具名 + 读音 + 英文名 / 等级 / 种类），目标处放射细线
  circle   召唤阵：人物脚下透视椭圆（三环 + 刻度 + 六芒星，慢转），底部一行"SERVANT · 职阶 · 真名"
合成沿用 v3：后景层（准线 / 圆 / 蜂巢 / 召唤阵）世界锁定 + 被前景人物遮挡 + 景深；前景层（文字面板）屏幕空间；
GPU 着色器里统一调光、泛光、运动模糊、glitch。
"""
import json, math, os
import numpy as np
import cv2
from PIL import Image, ImageDraw

import style_frames as SF
from style_frames import W, H, R, font
from cards import eo, bo, cl
import hud_elems as HE
import v3_hex as HX
from hud_v2 import Canvas, WHITE, GREY, F_WIDE, F_MONO, SC, SH, _c, _t
from hud_v3 import premult, glitch_rows

ROOT = r"D:\A\provant"
INFO = {k: v for k, v in json.load(open(os.path.join(ROOT, "data", "servant_info.json"), encoding="utf-8")).items() if not k.startswith("_")}
WHO = [("冲田总司〔Alter〕|冲田〔Alter〕", "okita_alter"), ("冲田总司", "okita"), ("宫本武藏", "musashi"), ("Saber Alter", "artoria_alter"),
       ("阿尔托莉雅|Saber$|^Saber|着甲Saber", "artoria"), ("贞德", "jeanne"), ("^Rider", "medusa"), ("迦尔纳", "karna"),
       ("莫德雷德", "mordred"), ("玛修", "mash"), ("金发持剑青年|红披风剑士|金发红披风", "richard"),
       ("伊什塔尔", "ishtar"), ("阿拉什", "arash"), ("贝德维尔", "bedivere"), ("狮子王", "lion_king"), ("魁札尔", "quetzalcoatl"), ("希波吕忒", "hippolyta")]
WORK = {"FSF": "FATE/STRANGE FAKE", "FGO": "FATE/GRAND ORDER", "HFM": "FATE/STAY NIGHT [HEAVEN'S FEEL]", "APO": "FATE/APOCRYPHA"}
DARK = (12, 12, 14)
TEMPLATES = ("scope", "frame", "hexwall", "status", "np", "circle")


def info_for(who, title, tags):
    """镜头的人物信息：servant_info 里有的用核对过的全量；没有的只用名字和职阶（v2 的 CHAR 表里那些参数 / 射程 / 回路数字不再上屏）。"""
    import re
    first = re.split(r"[、,，；;/+]", who or "")[0].strip()
    for pat, key in WHO:
        if re.search(pat, first):
            return dict(INFO[key], key=key)
    if title in ("UNIDENTIFIED", "", None):
        return None
    cls = next((t.split(None, 1)[1].strip() for t in tags if t.startswith("CLASS")), "")
    np_ = next((t.split(None, 1)[1].strip() for t in tags if t.startswith("NP")), "")
    if not any(ch.isalpha() for ch in cls):                 # "------" 之类的占位不上屏
        cls = ""
    if not cls and not np_:
        return dict(en=title, jp="", cls="", p=None, np_en="", np_jp="", np_ruby="", np_rank="", np_type="", key=None, label_only=True)
    if cls.startswith("BEAST"):
        np_ = ""                                            # 提亚马特等：表里写的不是宝具
    return dict(en=title, jp="", cls=cls, p=None, np_en=np_, np_jp="", np_ruby="", np_rank="", np_type="", key=None)


def rank_v(r):
    if not r or r[0] not in "EDCBA":
        return 6.0 if r and r.startswith("EX") else 0.0
    return "EDCBA".index(r[0]) + 1 + 0.25 * r.count("+")


def ell(cv, c, rx, ry, rot, a0, a1, col, a=1.0, th=1.0):
    if rx > 0.5 and ry > 0.5:
        cv2.ellipse(cv.a, (int(round(c[0] * SC)), int(round(c[1] * SC))), (int(rx * SC), int(ry * SC)), rot, a0, a1, _c(col, a), _t(th),
                    cv2.LINE_AA, SH)


_FC = {}


def jfont(size, weight="Bold"):
    k = (int(size), weight)
    if k not in _FC:
        _FC[k] = SF.vfont(SF.F_SERIF_CJK, int(size), weight)
    return _FC[k]


def text_w(s, f):
    return ImageDraw.Draw(Image.new("L", (1, 1))).textlength(s, font=f)


class HudV4:
    def __init__(self, seed, kind, info, work="", depth=None):
        rng = np.random.default_rng(seed * 7 + 211)
        self.seed, self.kind, self.info, self.work = seed, kind, info, work
        self.depth = float(depth if depth is not None else rng.uniform(0.70, 0.95))
        self.side = None

    # ------------------------------------------------------------ 共用
    def _name_block(self, cv, x, y, f, acc, align="l"):
        """真名块：英文宽体 + 日文明朝 + 职阶；逐行出。"""
        inf = self.info
        if not inf:
            return
        fe, fj = font(F_WIDE, 30 * R), jfont(30 * R)
        rows = [(inf["en"], fe, WHITE, 0), (inf.get("jp", ""), fj, WHITE, 5), (("CLASS  " + inf["cls"]) if inf.get("cls") else "", font(F_MONO, 18 * R), acc["hi"], 9)]
        yy = y
        for s, fnt, col, fk in rows:
            if not s:
                continue
            if f >= fk:
                k = eo((f - fk) / 8.0)
                w_ = text_w(s, fnt)
                xx = x if align == "l" else x - w_
                cv.d.text((xx + (1 - k) * 30 * R * (1 if align == "l" else -1), yy), s, font=fnt, fill=tuple(col) + (int(255 * k),))
            yy += fnt.size * 1.35
        if f >= 3:                                          # 强调色短横
            L_ = 120 * R * eo((f - 3) / 10.0)
            x0 = x if align == "l" else x - L_
            cv.rect((x0, y - 12 * R), (x0 + L_, y - 9 * R), acc["main"], 1.0)

    def _far_side(self, tc):
        return "r" if tc[0] < W * 0.5 else "l"

    # ------------------------------------------------------------ 模板
    def t_scope(self, back, front, f, tc, box, acc, bk, bf):
        cx, cy = tc
        r = float(np.clip(0.55 * max(box[2] - box[0], box[3] - box[1]), 0.18 * H, 0.34 * H))
        k = eo(f / 14.0)
        gap = r * 1.08
        for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # 全幅准线：从圆外向画边伸出
            far = (W if dx > 0 else 0) if dx else (H if dy > 0 else 0)
            L_ = abs(far - (cx if dx else cy)) - gap
            if L_ > 0:
                p0 = (cx + dx * gap, cy + dy * gap)
                back.line(p0, (p0[0] + dx * L_ * k, p0[1] + dy * L_ * k), WHITE, 0.55, 1.2 * R)
        back.circle((cx, cy), r * (0.6 + 0.4 * k), WHITE, 0.85, 1.6 * R)
        back.arc((cx, cy), r * (0.6 + 0.4 * k) + 9 * R, -30 + f * 0.6, 30 + f * 0.6, acc["main"], 1.0, 4 * R)
        for ang in range(0, 360, 90):                       # 4 个内刻度
            a_ = math.radians(ang)
            back.line((cx + math.cos(a_) * r * 0.86, cy + math.sin(a_) * r * 0.86), (cx + math.cos(a_) * r, cy + math.sin(a_) * r), WHITE, 0.9, 2 * R)
        # 边缘标尺（屏幕空间）：上 / 左，三角标指着目标坐标
        e = 38 * R
        for x in range(0, W, int(40 * R)):
            big = (x // int(40 * R)) % 5 == 0
            front.line((x, e), (x, e + (14 if big else 6) * R * k), WHITE, 0.5 if not big else 0.8, 1.0 * R)
        for y in range(int(80 * R), H, int(40 * R)):
            big = (y // int(40 * R)) % 5 == 0
            front.line((e, y), (e + (14 if big else 6) * R * k, y), WHITE, 0.5 if not big else 0.8, 1.0 * R)
        if f >= 6:
            front.tri([(cx, e + 18 * R), (cx - 8 * R, e + 32 * R), (cx + 8 * R, e + 32 * R)], acc["main"], 1.0)
            front.tri([(e + 18 * R, cy), (e + 32 * R, cy - 8 * R), (e + 32 * R, cy + 8 * R)], acc["main"], 1.0)
        side = self._far_side(tc)
        self._name_block(front, (W - 90 * R) if side == "r" else 90 * R, H * 0.70, f - 4, acc, align="r" if side == "r" else "l")

    def t_frame(self, back, front, f, tc, box, acc, bk, bf):
        m = 56 * R; L_ = 170 * R; th = 3.2 * R
        k = eo(f / 12.0); off = (1 - k) * 90 * R
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):  # 画幅四角大括号（从外滑入）
            x = (m if sx < 0 else W - m) + sx * off; y = (m if sy < 0 else H - m) + sy * off
            front.line((x, y), (x - sx * L_, y), WHITE, 0.95, th)
            front.line((x, y), (x, y - sy * L_ * 0.7), WHITE, 0.95, th)
        if self.work and f >= 4:
            front.text((m + 26 * R, m + 22 * R), self.work, 18 * R, WHITE, 0.9 * eo((f - 4) / 8.0), fnt=F_WIDE)
            front.rect((m + 26 * R, m + 52 * R), (m + 26 * R + 60 * R * eo((f - 6) / 8.0), m + 55 * R), acc["main"], 1.0)
        d = 14 * R                                          # 目标：一个小菱形 + 四短刻
        cx, cy = tc
        HE.poly(back, [(cx, cy - d), (cx + d, cy), (cx, cy + d), (cx - d, cy)], WHITE, 0.95, 1.6 * R)
        for a_ in (0, 90, 180, 270):
            u = math.radians(a_)
            back.line((cx + math.cos(u) * d * 1.8, cy + math.sin(u) * d * 1.8), (cx + math.cos(u) * d * 3.0, cy + math.sin(u) * d * 3.0), WHITE, 0.8, 1.4 * R)
        inf = self.info
        if not inf:
            return
        y0, y1 = H - m - 150 * R, H - m - 30 * R             # 底部全宽数据条（左 → 右擦出）
        x0, x1 = m + 26 * R, W - m - 26 * R
        wk = eo((f - 5) / 12.0)
        if wk <= 0:
            return
        xe = x0 + (x1 - x0) * wk
        front.rect((x0, y0), (xe, y1), DARK, 0.55)
        front.rect((x0, y0), (xe, y0 + 2.5 * R), acc["main"], 1.0)
        fields = [("CLASS", inf.get("cls", ""), ""), ("TRUE NAME", inf["en"], inf.get("jp", "")),
                  ("NOBLE PHANTASM", inf.get("np_en", ""), inf.get("np_jp", ""))]
        fields = [q for q in fields if q[1]]
        cw = (x1 - x0) / max(1, len(fields))
        for j, (lab, v1, v2) in enumerate(fields):
            fx = x0 + j * cw + 24 * R
            if fx > xe - 40 * R:
                break
            fk = 8 + 4 * j
            a_ = eo((f - fk) / 8.0)
            front.text((fx, y0 + 12 * R), lab, 13 * R, acc["hi"], a_, fnt=F_WIDE)
            fv = font(SF.F_BODONI_BLK, 34 * R) if j == 0 else font(F_WIDE, 22 * R)
            if j and text_w(v1, fv) > cw - 60 * R:            # 长宝具名：缩字号塞进本栏
                fv = font(F_WIDE, max(12, int(22 * R * (cw - 60 * R) / text_w(v1, fv))))
            front.d.text((fx, y0 + 34 * R), v1, font=fv, fill=WHITE + (int(255 * a_),))
            if v2:
                front.d.text((fx, y0 + 72 * R), v2, font=jfont(26 * R), fill=WHITE + (int(225 * a_),))
            if j:
                front.line((x0 + j * cw, y0 + 14 * R), (x0 + j * cw, y1 - 14 * R), WHITE, 0.4 * a_, 1.0 * R)

    def t_hexwall(self, back, front, f, tc, box, acc, bk, bf):
        side = self._far_side(tc)
        r = 74 * R
        dx, dy = 1.5 * r, math.sqrt(3) * r
        inf = self.info or {}
        words = [w for w in (inf.get("cls", ""), ("NP " + inf["np_rank"]) if inf.get("np_rank") and inf["np_rank"] != "—" else "",
                             "宝具" if inf.get("np_jp") else "", "真名" if inf.get("jp") else "") if w]
        rng = np.random.default_rng(self.seed)
        cells = []
        q = 0
        x = -r
        while x < W + r:
            frac = x / W
            far = frac if side == "r" else 1 - frac          # 0 = 人物那侧，1 = 远侧画边
            if far >= 0.46:
                for k_ in range(int(H / dy) + 2):
                    y = k_ * dy + (q % 2) * dy / 2 - dy / 2
                    if rng.random() < 0.30 + 0.70 * cl((far - 0.46) / 0.40):
                        cells.append((x, y, far))
            x += dx; q += 1
        if not cells:
            return
        n_fill = min(len(cells), max(2, len(words)) + 2)
        filled = list(int(i) for i in rng.choice(len(cells), n_fill, replace=False))
        label = {c_: words[j] for j, c_ in enumerate(filled[:len(words)])}
        for idx, (x, y, far) in enumerate(cells):
            fk = 1 + (far - 0.46) * 40 + rng.random() * 2   # 从靠人物那侧往画边一波弹出
            if f < fk:
                continue
            sc = 0.4 + 0.6 * bo(min(1.0, (f - fk) / 5.0), 2.0)
            pts = HX.hex_pts((x, y), r * 0.92, sc)
            if idx in filled:
                HE.fpoly(back, pts, acc["main"], 0.85)
                w_ = label.get(idx)
                if w_ and sc > 0.9:
                    im = HX.cond_text(w_, 30 * R if len(w_) <= 3 else 24 * R, (12, 10, 10))
                    back.txt.alpha_composite(im, (int(x - im.width / 2), int(y - im.height / 2)))
            else:
                HE.poly(back, pts, WHITE, 0.55, 1.3 * R)
        if inf.get("en") and f >= 6:                         # 顶部真名条（切角）
            k = eo((f - 6) / 10.0)
            tw = (380 * R + text_w(inf["en"], font(F_WIDE, 24 * R))) * k
            bx0 = (W - 70 * R - tw) if side == "r" else 70 * R
            by0 = 70 * R
            HE.fpoly(front, HE.chamfer_rect(bx0, by0, bx0 + tw, by0 + 56 * R, 14 * R, (1, 0, 1, 0)), DARK, 0.75)
            HE.poly(front, HE.chamfer_rect(bx0, by0, bx0 + tw, by0 + 56 * R, 14 * R, (1, 0, 1, 0)), acc["main"], 1.0, 1.6 * R)
            if k > 0.95:
                front.text((bx0 + 24 * R, by0 + 14 * R), "SERVANT  //  " + inf["en"], 24 * R, WHITE, 1.0, fnt=F_WIDE)
                if inf.get("jp"):
                    front.d.text((bx0 + 24 * R, by0 + 66 * R), inf["jp"], font=jfont(30 * R), fill=WHITE + (235,))

    def t_status(self, back, front, f, tc, box, acc, bk, bf):
        inf = self.info
        if not inf or not inf.get("p"):
            return self.t_frame(back, front, f, tc, box, acc, bk, bf)
        side = self._far_side(tc)
        pw, ph = 0.31 * W, 0.74 * H
        x0 = (W - 80 * R - pw) if side == "r" else 80 * R
        y0 = H * 0.13
        k = eo(f / 10.0)
        xs = x0 + (1 - k) * (60 * R if side == "r" else -60 * R)
        front.rect((xs, y0), (xs + pw, y0 + ph), DARK, 0.50 * k)
        front.rect((xs, y0), (xs + pw, y0 + ph), WHITE, 0.6 * k, 1.2 * R)
        front.rect((xs, y0), (xs + 6 * R, y0 + ph * eo(f / 14.0)), acc["main"], 1.0)
        px = xs + 40 * R
        if f >= 3:
            a_ = eo((f - 3) / 8.0)
            front.text((px, y0 + 30 * R), "CLASS", 16 * R, acc["hi"], a_, fnt=F_WIDE)
            front.d.text((px, y0 + 54 * R), inf["cls"], font=font(SF.F_BODONI_BLK, 72 * R), fill=WHITE + (int(255 * a_),))
        if f >= 6:
            a_ = eo((f - 6) / 8.0)
            front.text((px, y0 + 160 * R), inf["en"], 22 * R, WHITE, a_, fnt=F_WIDE)
            if inf.get("jp"):
                front.d.text((px, y0 + 192 * R), inf["jp"], font=jfont(30 * R), fill=WHITE + (int(240 * a_),))
        yy = y0 + 262 * R
        front.line((px, yy - 14 * R), (xs + pw - 40 * R, yy - 14 * R), WHITE, 0.45, 1.0 * R)
        gx0, gx1 = px + 150 * R, xs + pw - 120 * R
        for j, key in enumerate(("STR", "END", "AGI", "MAN", "LUK", "NP")):
            fk = 9 + 3 * j
            if f < fk:
                break
            a_ = eo((f - fk) / 6.0)
            rk = inf["p"].get(key, "")
            ry = yy + j * 58 * R
            front.text((px, ry + 8 * R), key, 22 * R, WHITE, a_, fnt=F_WIDE)
            gl = (gx1 - gx0) * rank_v(rk) / 6.0 * eo((f - fk) / 14.0)
            front.rect((gx0, ry + 14 * R), (gx1, ry + 30 * R), WHITE, 0.25 * a_, 1.0 * R)
            if gl > 1:
                front.rect((gx0, ry + 14 * R), (gx0 + gl, ry + 30 * R), acc["main"], 0.95 * a_)
            front.d.text((gx1 + 24 * R, ry - 4 * R), rk, font=font(SF.F_BODONI_BLK, 40 * R), fill=WHITE + (int(255 * a_),))
        if inf.get("np_jp") and f >= 28:                    # 宝具：日文名 + 读音
            a_ = eo((f - 28) / 8.0)
            ny = yy + 6 * 58 * R + 18 * R
            front.text((px, ny), "NOBLE PHANTASM", 14 * R, acc["hi"], a_, fnt=F_WIDE)
            if inf.get("np_ruby"):
                front.d.text((px, ny + 24 * R), inf["np_ruby"], font=jfont(18 * R, "Regular"), fill=WHITE + (int(210 * a_),))
            front.d.text((px, ny + 46 * R), inf["np_jp"], font=jfont(36 * R, "Black"), fill=WHITE + (int(255 * a_),))
            meta = "  ".join(x for x in (("RANK " + inf["np_rank"]) if inf.get("np_rank") else "", inf.get("np_type", "")) if x)
            front.text((px, ny + 98 * R), meta, 15 * R, WHITE, 0.85 * a_, fnt=F_WIDE)
        cx, cy = tc                                          # 一根引线：目标 → 面板
        ex = xs if side == "r" else xs + pw
        L_ = eo((f - 4) / 10.0)
        if L_ > 0:
            back.line((cx, cy), (cx + (ex - cx) * L_, cy + (y0 + 60 * R - cy) * L_), WHITE, 0.6, 1.2 * R)
            back.circle((cx, cy), 10 * R, acc["main"], 1.0, 2 * R)

    def t_np(self, back, front, f, tc, box, acc, bk, bf):
        inf = self.info
        if not inf or not (inf.get("np_jp") or inf.get("np_en")):
            return self.t_scope(back, front, f, tc, box, acc, bk, bf)
        cx, cy = tc
        rng = np.random.default_rng(self.seed)
        for j in range(14):                                 # 目标处放射细线
            a_ = rng.uniform(0, 2 * math.pi)
            r0 = 0.20 * H
            L_ = (0.25 + rng.random() * 0.6) * W * eo((f - j % 4) / 10.0)
            if L_ > 0:
                back.line((cx + math.cos(a_) * r0, cy + math.sin(a_) * r0), (cx + math.cos(a_) * (r0 + L_), cy + math.sin(a_) * (r0 + L_)),
                          WHITE, 0.45, 1.3 * R)
        bh = int(168 * R)
        band = Image.new("RGBA", (int(W * 1.3), bh), (0, 0, 0, 0)); d = ImageDraw.Draw(band)
        d.rectangle([0, 0, band.width, band.height], fill=DARK + (150,))
        d.rectangle([0, 0, band.width, int(3 * R)], fill=tuple(acc["main"]) + (255,))
        d.rectangle([0, band.height - int(3 * R), band.width, band.height], fill=tuple(acc["main"]) + (255,))
        bx = int(W * 0.15 + 60 * R)
        name = inf.get("np_jp") or inf["np_en"]
        fj = jfont(92 * R, "Black") if inf.get("np_jp") else font(F_WIDE, 64 * R)
        nshow = max(0, min(len(name), int((f - 6) / 1.5)))
        if inf.get("np_ruby") and f >= 6:
            d.text((bx, 4 * R), inf["np_ruby"], font=jfont(22 * R, "Regular"), fill=WHITE + (220,))
        d.text((bx, 28 * R), name[:nshow], font=fj, fill=WHITE + (255,))
        if f >= 14:
            a_ = int(255 * eo((f - 14) / 8.0))
            meta = "  ".join(x for x in (inf.get("np_en", "") if inf.get("np_jp") else "", ("RANK " + inf["np_rank"]) if inf.get("np_rank") else "",
                                          inf.get("np_type", "")) if x)
            tx = bx + text_w(name, fj) + 40 * R
            d.text((tx, 52 * R), meta, font=font(F_WIDE, 20 * R), fill=WHITE + (a_,))
            d.text((tx, 92 * R), inf["en"] + (("  /  " + inf["cls"]) if inf.get("cls") else ""), font=font(F_MONO, 16 * R),
                   fill=tuple(acc["hi"]) + (a_,))
        band = band.rotate(7, resample=Image.BICUBIC, expand=True)
        k = eo(f / 9.0)
        by = (H * 0.72 if cy < H * 0.55 else H * 0.24) - band.height / 2
        bxo = -W * 0.15 - (1 - k) * W * 0.7
        front.txt.alpha_composite(band, (int(bxo), int(by)))

    def t_circle(self, back, front, f, tc, box, acc, bk, bf):
        x0, y0, x1, y1 = box
        foot = ((x0 + x1) / 2, min(H * 0.92, y1 - 0.02 * H))
        rx = float(np.clip(1.15 * (x1 - x0), 0.30 * W, 0.46 * W)); sq = 0.30
        g = eo(f / 16.0); rot = f * 0.35
        if g <= 0:
            return
        for s_, th, al in ((1.0, 2.4, 0.95), (0.92, 1.2, 0.8), (0.62, 1.6, 0.85)):     # 三环
            ell(back, foot, rx * s_ * g, rx * s_ * sq * g, 0, 0, 360, WHITE, al, th * R)
        for j in range(72):                                 # 一二环之间的刻度（慢转）
            a_ = math.radians(j * 5 + rot)
            r_in, r_out = (0.93 if j % 6 else 0.86), 0.99
            p = (foot[0] + math.cos(a_) * rx * r_in * g, foot[1] + math.sin(a_) * rx * r_in * sq * g)
            q = (foot[0] + math.cos(a_) * rx * r_out * g, foot[1] + math.sin(a_) * rx * r_out * sq * g)
            back.line(p, q, WHITE, 0.7, 1.0 * R)
        for tri_off in (0, 60):                             # 六芒星（透视压扁、反向慢转）
            pts = []
            for j in range(3):
                a_ = math.radians(tri_off + j * 120 - rot * 0.6 - 90)
                pts.append((foot[0] + math.cos(a_) * rx * 0.62 * g, foot[1] + math.sin(a_) * rx * 0.62 * sq * g))
            HE.poly(back, pts, acc["main"], 0.95, 2.0 * R)
        ell(back, foot, rx * 1.06 * g, rx * 1.06 * sq * g, 0, rot, rot + 50, acc["main"], 1.0, 4 * R)
        inf = self.info
        if inf and f >= 10:
            a_ = eo((f - 10) / 10.0)
            s = "SERVANT   ·   " + (inf.get("cls") or "") + "   ·   " + inf["en"]
            fnt = font(F_WIDE, 26 * R)
            front.d.text((W / 2 - text_w(s, fnt) / 2, 96 * R), s, font=fnt, fill=WHITE + (int(255 * a_),))      # 放顶部：不压在召唤阵上
            if inf.get("jp"):
                fj = jfont(26 * R)
                front.d.text((W / 2 - text_w(inf["jp"], fj) / 2, 138 * R), inf["jp"], font=fj, fill=WHITE + (int(220 * a_),))

    # ------------------------------------------------------------ 采样放大窗（scope 用；用户："hud 和采样框结合的可以"）
    def sample(self, base, src, f, tc, side, acc):
        if src is None or f < 8:
            return base
        cx, cy = tc
        sw, sh_ = 260 * R, 146 * R                          # 源处采样框
        sx0 = int(np.clip(cx - sw / 2, 0, W - sw)); sy0 = int(np.clip(cy - sh_ / 2, 0, H - sh_))
        ww, wh = int(W * 0.30), int(W * 0.30 * 9 / 16)       # 放大窗（2.95×）
        wx0 = int(W - 90 * R - ww) if side == "r" else int(90 * R)
        wy0 = int(H * 0.16)
        k = eo((f - 8) / 8.0)
        crop = cv2.resize(src[sy0:sy0 + int(sh_), sx0:sx0 + int(sw)], (ww, max(1, int(wh * k))), interpolation=cv2.INTER_CUBIC)
        out = base.copy()
        out[wy0:wy0 + crop.shape[0], wx0:wx0 + ww] = crop
        cv2.rectangle(out, (wx0, wy0), (wx0 + ww, wy0 + crop.shape[0]), WHITE, max(1, int(1.5 * R)), cv2.LINE_AA)
        cv2.rectangle(out, (sx0, sy0), (sx0 + int(sw), sy0 + int(sh_)), tuple(int(v) for v in acc["main"]), max(1, int(1.5 * R)), cv2.LINE_AA)
        p0 = (sx0 + int(sw), sy0) if side == "r" else (sx0, sy0)
        p1 = (wx0, wy0) if side == "r" else (wx0 + ww, wy0)
        cv2.line(out, p0, p1, WHITE, 1, cv2.LINE_AA)
        cv2.putText(out, "ZOOM x2.9", (wx0 + 12, wy0 + crop.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * R, WHITE, 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------ 合成（同 v3：后景世界锁定 + 遮挡；前景屏幕空间）
    def render3(self, base, f, t, bk, bf, box, acc, rms, src=None, life_in=1.0, life_out=0.0, cam=None, cam_vel=(0.0, 0.0),
                fg=None, n_samples=1, burst=False, hud_glitch=0.0, gseed=0, kind="face"):
        if life_out >= 1.0:
            return base
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        if bw > 0.40 * W or bh > 0.60 * H:                   # 特写：瞄眼睛一带
            tc = ((x0 + x1) / 2, y0 + (0.40 if kind != "person" else 0.13) * bh)
        else:
            tc = ((x0 + x1) / 2, y0 + 0.40 * bh)
        tc = (float(np.clip(tc[0], 0.12 * W, 0.88 * W)), float(np.clip(tc[1], 0.14 * H, 0.86 * H)))
        back, front = Canvas(), Canvas()
        getattr(self, "t_" + self.kind)(back, front, f, tc, box, acc, bk, bf)
        out = base
        if self.kind == "scope" and n_samples and life_out <= 0.0:
            out = self.sample(base, src, f, tc, self._far_side(tc), acc)
        import gpu
        ax, ay = tc
        M = np.eye(3, dtype=np.float64)
        if life_out > 0:
            sx = 1.0 + 0.08 * life_out
            M = np.array([[sx, 0, ax * (1 - sx)], [0, sx, ay * (1 - sx)], [0, 0, 1]]) @ M
        if cam is not None:
            C = np.eye(3); C[:2] = np.asarray(cam, np.float64)
            M = (np.eye(3) + self.depth * (C - np.eye(3))) @ M
        rad = eo(life_in) * 1.25 * math.hypot(W, H) if life_in < 1.0 else 1e6
        g = max(hud_glitch, 0.8 * life_out)
        if gpu.use():
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
            return gpu.hud3(out, back.a, np.asarray(back.txt), front.a, np.asarray(front.txt), np.linalg.inv(M), (ax, ay), rad, 80 * R,
                            1.0 - life_out, 1.0 - life_out, fg=fgu, rows=rows, vel=cam_vel, rgbsplit=10 * R * g if g > 0.02 else 0.0)
        Pb, Pf = premult(back), premult(front)
        if cam is not None or life_out > 0:
            Pb = cv2.warpAffine(Pb, M[:2].astype(np.float32), (W, H))
        if fg is not None:
            Pb *= (1 - fg)[..., None]
        P = Pf * (1 - life_out) + Pb * (1 - Pf[..., 3:4]) * (1 - life_out)
        if g > 0.02:
            P = glitch_rows(P, g, gseed)
        o = out.astype(np.float32) / 255 * (1 - P[..., 3:4]) + P[..., :3]
        return (np.clip(o, 0, 1) * 255).astype(np.uint8)

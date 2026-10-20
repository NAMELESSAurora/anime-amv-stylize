# -*- coding: utf-8 -*-
"""
v2 风格预览片（给用户看方向用，不进成片）。两段：

A  转场 / 风格样片，歌曲 20.9–30.0 s（歌词 8–12）：
   S1 R  Saber Alter 面甲 ──glitch──▶ S2 R Lancer 怒吼 ──M4 HUD 锁定接力──▶ S3 C暗 贞德对镜 星座线
   ──M9 档案拍贴──▶ S4 C亮 齐格 + 贞德 档案 ──glitch + 砸入──▶ S5 R Lancer 举枪 + 重拍波形撕裂
   （初版 T2/T3 是 splash，用户看过说"太粗糙了"、"可以设计些别的特效或者 motion 来替换"，已换掉。）
B  高潮样片，歌曲 124.2–132.214 s。用户 10-18："高潮要配合 grokbot 整理的三种风格加上 glitch 来营造情绪张力和爆点"。
   量过：127.570 s 能量骤降（换气），128.128 s 全曲最大跃升（+0.148 RMS，E11 末段高能量平台起点）= 爆点。
   蓄力 124.2–127.57：每拍一切，R（Raf0x00 HUD）/ C（causasui 档案）/ P（hectoroz 版画 + HUD + 裁切框）轮换，
                      全局 glitch 强度随蓄力爬升；
   换气 127.57–128.128：爆点镜头的第一帧定格成版画，线条沸腾（hold 2）、取景框收紧、角落锈块蔓延、glitch 冲顶；
   爆点 128.128：3 帧版画冲击帧（glitch 满 + 波形撕裂）→ 砸入彩色，HUD 全部元素同时炸出；
   之后每拍一切 + 每刀 glitch，M4 / M9 收尾。

取景框 / 标题条 / 像素块 / 锁定角标 / 歌词竖条的颜色都跟各自画面的主色走（v2_looks.AccentTrack）；
HUD 用 hud_v2.HudKit（照 Raf0x00 原帖的气泡簇 / 三档线宽 / 斜纹盘 / 光栅条 / 刻度尺）。
流程：analyze（逐镜头顺序解码：强调色、抠像、锚点——R/P 光流跟框，C 每 4 帧检测脸再插值平滑）
      → render（多进程逐帧 PNG）→ encode（NVENC + 原曲）→ sheet。
用法: python scripts/v2_preview.py A|B [analyze|render|encode|sheet|all]
"""
import os, sys, json, math, pickle, subprocess, time
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(__file__))
import style_frames as SF
from style_frames import W, H, R, font
from cards import LiveSource, eo, bo, cl
import v2_looks as L
import v2_trans as T
from hud_v2 import HudKit
from hud_v3 import HudV3, spring_track
import v3_manga as MG
import v3_hex as HX
import hud_v4 as H4
import v5_tape as TP
import v6_title as TT
import palette as PAL
import v4_looks as V4
import v7_cine as CN

ROOT = SF.ROOT
FPS = 60
GRID = json.load(open(os.path.join(ROOT, "data", "pulse_grid_adaptive.json")))
BEATS = GRID["beat_times_s"]
LYR = json.load(open(L.LYR, encoding="utf-8"))["rows"]
CALL = [("NP-DRIVER ->", 1, -1.1, False), ("MANA 967MiB / 2048MiB", 1.1, 0.9, False), ("SERVANT LINK 20:02", -1.2, 0.6, True)]

# lo / hi = 源镜头可用范围（量过切点）；超出停在边界帧。speed 0 = 定格；hold = 步进取帧（fps）
SA_TAGS = ["CLASS  SABER [ALTER]", "PARAM  STR A / MGI A++", "NP     EXCALIBUR MORGAN"]
LA_TAGS = ["CLASS  LANCER", "NP     GAE BOLG", "RANGE  2-4 / 1"]
RI_TAGS = ["CLASS  SABER", "TRUE   RICHARD I", "NP     EXCALIBUR"]
JE_TAGS = ["CLASS  RULER", "TRUE   JEANNE D'ARC", "NP     LUMINOSITE ETERNELLE"]
JE_LAB = ["RULER", "LA PUCELLE", "BANNER", "ORLEANS", "1412", "FAITH", "MIRROR", "SAINT"]

CFG = {
    "A": dict(
        T0=20.9, T1=30.0, out="preview_v2_A.mp4",
        shots=[
            dict(id="S1", look="R", yid="fKoHcAM0PEk", a=20.9, b=22.268, src_a=15.80, speed=1.0, lo=15.70, hi=18.85,
                 det=("face", "head"), manual=(1040, 250, 1580, 900), title="SABER ALTER", tags=SA_TAGS),
            dict(id="S2", look="R", yid="5IiGdXV-wow", a=22.268, b=23.429, src_a=35.55, speed=1.0, lo=35.42, hi=37.55,
                 det=("face", "head"), manual=(700, 300, 1900, 1300), title="LANCER", tags=LA_TAGS),
            dict(id="S3", look="Cd", yid="CQ8v15KMND4", a=23.429, b=25.774, src_a=44.78, speed=0.8, lo=44.64, hi=46.84,
                 keep_x=1230, labels=JE_LAB),
            dict(id="S4", look="Cb", yid="CQ8v15KMND4", a=25.774, b=28.096, src_a=87.62, speed=1.0, lo=87.33, hi=90.08,
                 titles=["SIEG // HOMUNCULUS", "RULER // JEANNE D'ARC"], box_beats=[26.331, 26.935]),
            dict(id="S5", look="R", yid="5IiGdXV-wow", a=28.096, b=30.0, src_a=44.12, speed=0.94, lo=44.02, hi=45.91,
                 det=("person",), manual=(1500, 300, 2050, 1440), title="LANCER", tags=["NP     GAE BOLG", "MODE   THROW", "LOCK   SOUL"],
                 tear=[29.211]),
        ],
        trans=[dict(t=22.268, kind="glitch", pre=6, post=6), dict(t=23.429, kind="lock_zoom", pre=12, post=6),
               dict(t=25.774, kind="panel_slap", pre=12, post=12), dict(t=28.096, kind="glitch_slam", pre=6, post=8)],
        lyrics={8: ("hud", (0.60, 0.70)), 9: ("burst", None), 10: ("serif_dark", None), 11: ("serif_ink", None), 12: ("burst", None)},
        glitch=[],
    ),
    "B": dict(
        T0=124.2, T1=132.214, out="preview_v2_B_climax.mp4",
        shots=[
            # ---- 蓄力：每拍一切
            dict(id="B1", look="R", yid="fKoHcAM0PEk", a=124.204, b=124.807, src_a=15.80, speed=1.0, lo=15.70, hi=18.85,
                 det=("face", "head"), manual=(1040, 250, 1580, 900), title="SABER ALTER", tags=SA_TAGS),
            dict(id="B2", look="Cd", yid="CQ8v15KMND4", a=124.807, b=125.388, src_a=44.85, speed=1.0, lo=44.64, hi=46.84,
                 keep_x=1230, labels=JE_LAB),
            dict(id="B3", look="R", yid="5IiGdXV-wow", a=125.388, b=125.968, src_a=35.60, speed=1.0, lo=35.42, hi=37.55,
                 det=("face", "head"), manual=(700, 300, 1900, 1300), title="LANCER", tags=LA_TAGS),
            dict(id="B4", look="P", yid="fKoHcAM0PEk", a=125.968, b=126.549, src_a=119.60, speed=1.0, lo=119.42, hi=122.0, hold=12,
                 det=("face", "head"), manual=(900, 300, 1700, 1100), title="SABER ALTER", tags=SA_TAGS, cropbox="VISOR // ALTER"),
            dict(id="B5", look="R", yid="5Jyflcob9_k", a=126.549, b=127.060, src_a=63.22, speed=1.0, lo=63.21, hi=63.74,
                 det=("face", "head"), manual=(700, 200, 1900, 1300), title="RICHARD I", tags=RI_TAGS),
            dict(id="B6", look="P", yid="VobxMYXmkao", a=127.060, b=127.570, src_a=12.80, speed=1.0, lo=12.62, hi=15.2, hold=12,
                 det=("face", "head"), manual=(800, 200, 1800, 1200), title="JEANNE", tags=JE_TAGS, cropbox="RULER // EYES"),
            # ---- 换气：爆点镜头第一帧定格成版画
            dict(id="G", look="P", yid="fKoHcAM0PEk", a=127.570, b=128.128, src_a=127.30, speed=0.0, lo=127.04, hi=128.0,
                 det=("person", "head"), manual=(1000, 300, 1560, 1440), title="SABER ALTER", tags=SA_TAGS,
                 boil=True, hud_rate=0.42, growth=True, tension=True, push=0.08, bars=0.11),
            # ---- 爆点
            dict(id="D1", look="R", yid="fKoHcAM0PEk", a=128.128, b=128.708, src_a=127.30, speed=1.0, lo=127.04, hi=127.98,
                 det=("person", "head"), manual=(1000, 300, 1560, 1440), title="SABER ALTER", tags=SA_TAGS,
                 impact=3, slam=12, burst=True, tear=[128.128]),
            dict(id="D2", look="R", yid="5Jyflcob9_k", a=128.708, b=129.289, src_a=63.85, speed=1.0, lo=63.75, hi=64.66,
                 det=("person", "head"), manual=(1000, 200, 1600, 1440), title="RICHARD I", tags=RI_TAGS, burst=True),
            dict(id="D3", look="Cd", yid="VobxMYXmkao", a=129.289, b=129.869, src_a=13.90, speed=1.0, lo=12.62, hi=15.2,
                 labels=["RULER", "SAINT", "BANNER", "ORLEANS", "1412", "FAITH", "PETAL", "MOON"]),
            dict(id="D4", look="R", yid="5IiGdXV-wow", a=129.869, b=131.053, src_a=44.12, speed=1.0, lo=44.02, hi=45.91,
                 det=("person",), manual=(1500, 300, 2050, 1440), title="LANCER", tags=["NP     GAE BOLG", "MODE   THROW", "LOCK   SOUL"],
                 tear=[129.869], burst=True),
            dict(id="D5", look="Cb", yid="CQ8v15KMND4", a=131.053, b=132.214, src_a=87.62, speed=1.0, lo=87.33, hi=90.08,
                 titles=["SIEG // HOMUNCULUS", "RULER // JEANNE D'ARC"], box_beats=[131.634, 131.92]),
        ],
        trans=[dict(t=124.807, kind="gcut", pre=2, post=3), dict(t=125.388, kind="gcut", pre=2, post=3),
               dict(t=125.968, kind="gcut", pre=2, post=3), dict(t=126.549, kind="gcut", pre=2, post=3),
               dict(t=127.060, kind="gcut", pre=2, post=3), dict(t=127.570, kind="gcut", pre=2, post=2),
               dict(t=128.708, kind="glitch", pre=3, post=4), dict(t=129.289, kind="lock_zoom", pre=12, post=6),
               dict(t=129.869, kind="glitch_slam", pre=4, post=8), dict(t=131.053, kind="panel_slap", pre=12, post=12)],
        lyrics={54: ("hud", (0.06, 0.80)), 55: ("burst", None), 56: ("burst", None), 57: ("burst", None)},
        # 全局 glitch 包络：(t0, t1, a0, a1)，蓄力爬升 → 换气冲顶 → 爆点后 8 帧落回
        punch=[128.128],
        glitch=[(124.2, 127.0, 0.0, 0.22), (127.0, 127.57, 0.22, 0.35), (127.57, 128.128, 0.35, 0.9), (128.128, 128.27, 1.0, 0.0)],
    ),
}


def fr(t):
    return int(round(t * FPS))


_MAPPED = {"done": False}


def _apply_song_map(c):
    """v9 成片（edl_v11）：EDL 是成片时间；歌的拍点 / 歌词 / 逐词时间是歌曲时间 → 按 song_map 平移（每个进程只做一次）。"""
    global _LEND, _LW, EIGHTHS
    if _MAPPED["done"] or not c.get("song_map"):
        return
    segs = c["song_map"]                                    # [[成片起点, 歌起点, 歌终点], …]

    def s2f(t):
        for f0, a, b in segs:
            if a <= t < b:
                return f0 + (t - a)
        f0, a, b = segs[-1]
        return f0 + (t - a)
    def seg_of(t):
        return next((k for k, (f0, a, b) in enumerate(segs) if a - 1e-6 <= t < b - 1e-6), len(segs) - 1)

    def s2f_end(t0, t1):
        """结束时间：和开始不在同一段（跨过 break）就截到开始那段的末尾，不让一句歌词挂过整个 break。"""
        k0, k1 = seg_of(t0), seg_of(t1 - 1e-4)
        if k0 == k1:
            return s2f(t1)
        f0, a, b = segs[k0]
        return f0 + (b - a)
    BEATS[:] = [s2f(b) for b in BEATS]
    EIGHTHS = sorted(BEATS + [(a + b) / 2 for a, b in zip(BEATS, BEATS[1:])])
    st0 = [r["start"] / 1000.0 for r in LYR]
    for r in LYR:
        s0_ = r["start"] / 1000.0
        r["start"] = s2f(s0_) * 1000.0; r["end"] = s2f_end(s0_, r["end"] / 1000.0) * 1000.0
    p = os.path.join(ROOT, "data", "lyric_end.json")
    _LEND = {int(k): s2f_end(st0[int(k)], v) for k, v in json.load(open(p)).items()} if os.path.exists(p) else {}
    p = os.path.join(ROOT, "data", "lyric_words.json")
    _LW = {int(k): [[w, s2f(t)] for w, t in v] for k, v in json.load(open(p, encoding="utf-8")).items()} if os.path.exists(p) else {}
    _MAPPED["done"] = True


def prep(key):
    if key not in CFG:                                     # 成片：scripts/edl_v4.py 产出的 data/edl_v4.json
        CFG[key] = json.load(open(os.path.join(ROOT, "data", "edl_%s.json" % key.lower()), encoding="utf-8"))
        CFG[key]["lyrics"] = {int(k): v for k, v in CFG[key]["lyrics"].items()}
    c = CFG[key]
    c["key"] = key
    _apply_song_map(c)
    c["lyrics"] = {i: (v if isinstance(v, dict) else dict(style=v[0], pos=v[1])) for i, v in c["lyrics"].items()}
    c["F0"], c["F1"] = fr(c["T0"]), fr(c["T1"])
    c["work"] = os.path.join(ROOT, "media", "v2prev", key) if key in ("A", "B") else os.path.join(r"E:\provant_work", key)
    c["ana"] = os.path.join(c["work"], "ana"); c["frames"] = os.path.join(c["work"], "frames")
    for tr in c["trans"]:
        tr["F"] = fr(tr["t"])
    for s in c["shots"]:
        s["Fa"], s["Fb"] = fr(s["a"]), fr(s["b"])
        pre = next((tr["pre"] for tr in c["trans"] if tr["F"] == s["Fa"]), 0)
        post = next((tr["post"] for tr in c["trans"] if tr["F"] == s["Fb"]), 0)
        s["F_in"], s["F_out"] = max(c["F0"], s["Fa"] - pre), min(c["F1"], s["Fb"] + post)
    return c


def src_t(s, F):
    rel = F / FPS - s["a"]
    if s.get("hold"):                                      # 步进取帧（hectoroz 的"姿势沸腾"）
        rel = math.floor(rel * s["hold"]) / s["hold"]
    return float(np.clip(s["src_a"] + rel * s["speed"], s["lo"], s["hi"]))


def beat_state(F):
    t = F / FPS
    past = [b for b in BEATS if b <= t]
    return len(past), ((t - past[-1]) * FPS if past else 99)


def glitch_amp(c, F):
    t = F / FPS
    for t0, t1, a0, a1 in c["glitch"]:
        if t0 <= t < t1:
            return a0 + (a1 - a0) * (t - t0) / (t1 - t0)
    return 0.0


# ---- 逐词炸出的分词（用户 10-18："歌词可以出现的更炸裂一些，比如一个词一个词出，例如 GO, TAKE, SOME, COVER!"）
# 只改大小写 / 断词；标点照原文，唯一例外是用户自己示范的 "COVER!"。没列的句子按空格切、≤2 字母的词并进下一个。
BURST = {
    9: ["COME ON!", "SHOW ME", "HOW YOU", "THINK!"],
    12: ["JUST", "LIKE", "ALL", "THE", "REST?"],          # 慢唱拖长音（用户："原曲是慢慢说出这几个词的"），一词一出
    55: ["GO", "TAKE", "SOME", "COVER!"],
    56: ["BUTCHYA", "GOTTA", "BREAK OUT", "TO BE", "THE BRAT"],
    57: ["CUZ YA", "TIRED", "OF BEING", "JUST A", "COPYCAT"],
}
EIGHTHS = sorted(BEATS + [(a + b) / 2 for a, b in zip(BEATS, BEATS[1:])])


PART = "のをがにはでとへもや"
FUNC = {"THE", "AND", "HIS", "TO", "A", "OF", "IN", "BE", "CUZ", "YA", "YOU", "ALL", "NOT", "SO", "UP", "ON", "IT"}


def is_kanji(ch):
    return 0x4E00 <= ord(ch) <= 0x9FFF


def is_kana(ch):
    return 0x3040 <= ord(ch) <= 0x30FF


def jp_chunks(tok):
    """日文短语切块：助词后断、假名→汉字处断（送假名边界）、最长 5 字。超俗した / 次元の / 契約を / 刻む。"""
    out, cur = [], ""
    for i, ch in enumerate(tok):
        cur += ch
        nxt = tok[i + 1] if i + 1 < len(tok) else ""
        if not nxt:
            break
        if (ch in PART and len(cur) >= 2) or (is_kana(ch) and is_kanji(nxt) and len(cur) >= 3) or len(cur) >= 5:
            out.append(cur); cur = ""
    if cur:
        if len(cur) == 1 and out:
            out[-1] += cur
        else:
            out.append(cur)
    return out


def burst_words(i):
    if i in BURST:
        return BURST[i]
    out, carry = [], ""
    for w in LYR[i]["text"].replace("\uff1f", "?").split():
        if L.is_cjk(w[0]):
            if carry:
                out.append(carry); carry = ""
            out += jp_chunks(w)
            continue
        u = w.upper()
        if u in FUNC and carry.count(" ") < 1:             # 虚词挂到下一个实词上（最多两个虚词连挂）
            carry = (carry + " " + u).strip(); continue
        w = (carry + " " + u).strip() if carry else u
        out.append(w); carry = ""
    if carry:
        out.append(carry)
    return out


_LW = None
_LEND = None


def lyr_end(i):
    """句子显示结束时间：data/lyric_end.json 有覆盖就用（慢唱句字幕轴偏短），否则用字幕轴。"""
    global _LEND
    if _LEND is None:
        p = os.path.join(ROOT, "data", "lyric_end.json")
        _LEND = {int(k): v for k, v in json.load(open(p)).items()} if os.path.exists(p) else {}
    return _LEND.get(i, LYR[i]["end"] / 1000.0)


def burst_times(i):
    """词起点：优先用 data/lyric_words.json（scripts/lyric_time.py：分离干声的起音 + DP 对齐，用户 10-19"歌词跟拍需要优化"）；
    没有时退回旧法：句长前 85% 均分，吸附到八分音符网格。"""
    global _LW
    if _LW is None:
        p = os.path.join(ROOT, "data", "lyric_words.json")
        _LW = {int(k): v for k, v in json.load(open(p, encoding="utf-8")).items()} if os.path.exists(p) else {}
    if i in _LW and len(_LW[i]) == len(burst_words(i)):
        return [w for w, _ in _LW[i]], [t for _, t in _LW[i]]
    s0, e0 = LYR[i]["start"] / 1000.0, LYR[i]["end"] / 1000.0
    words = burst_words(i)
    ts, prev = [s0], s0
    for k in range(1, len(words)):
        ideal = s0 + k * 0.85 * (e0 - s0) / len(words)
        cand = [g for g in EIGHTHS if g >= prev + 0.12 and g < e0 - 0.1]
        t = min(cand, key=lambda g: abs(g - ideal)) if cand else ideal
        ts.append(t); prev = t
    return words, ts


# ---------------------------------------------------------------- analyze
class PtTrack:
    def __init__(self, g, pts):
        self.prev = g; self.p = np.float32(pts).reshape(-1, 1, 2) / 2


def box_track(frames_g, box0):
    """框跟踪：框内角点的中位位移推动框中心（每 15 帧在当前框内重新取点），再时间平滑。"""
    x0, y0, x1, y1 = box0
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    out = [(cx, cy)]
    prev = frames_g[0]; pts = None
    for i in range(1, len(frames_g)):
        if pts is None or i % 15 == 1 or len(pts) < 6:
            m = np.zeros_like(prev)
            cv2.rectangle(m, (int((cx - bw / 2) / 2), int((cy - bh / 2) / 2)), (int((cx + bw / 2) / 2), int((cy + bh / 2) / 2)), 255, -1)
            pts = cv2.goodFeaturesToTrack(prev, 80, 0.01, 6, mask=m)
        if pts is not None and len(pts):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, frames_g[i], pts, None, winSize=(31, 31), maxLevel=3)
            ok = st.reshape(-1) == 1
            if ok.sum() >= 3:
                d = np.median((nxt - pts).reshape(-1, 2)[ok], axis=0) * 2
                cx, cy = cx + d[0], cy + d[1]
                pts = nxt[ok].reshape(-1, 1, 2)
            else:
                pts = None
        out.append((cx, cy)); prev = frames_g[i]
    c = gaussian_filter1d(np.array(out), 3, axis=0, mode="nearest")
    return np.column_stack([c[:, 0] - bw / 2, c[:, 1] - bh / 2, c[:, 0] + bw / 2, c[:, 1] + bh / 2])


def pick_box(rgb, kinds, manual):
    from anime_det import Det
    for k in kinds:
        if k == "manual":                                  # v10：EDL 里手标的取景框（强光 / 多人 / 小人物，检测会锁错，用户："0:02:39 的 HUD 没有对准人物"）
            return tuple(manual), "manual"
        b = Det(k)(rgb, th=0.25)
        if b:
            return tuple(b[0][:4]), k
    return tuple(manual), "manual"


def vs_track(s, vsd, n):
    """对决牌两个目标的框 [n, 2, 4]：每个检测帧取面积最大的两个人物框按 x 排左右，最近邻续接（跳 > 25% 画宽视为丢失），缺帧插值、平滑；
    一个都没有 → 用 EDL 里的 vs.pts（画面比例坐标）。"""
    pts = (s.get("vs") or {}).get("pts") or [(0.35, 0.5), (0.65, 0.5)]
    dflt = [[(p[0] - (p[2] if len(p) > 2 else 0.12) / 2) * W, (p[1] - (p[3] if len(p) > 3 else 0.20) / 2) * H,
             (p[0] + (p[2] if len(p) > 2 else 0.12) / 2) * W, (p[1] + (p[3] if len(p) > 3 else 0.20) / 2) * H] for p in pts]   # pts：(cx, cy[, w, h]) 画面比例
    ks, bs = [], []
    if (s.get("vs") or {}).get("fixed"):                   # 强光 / 小人物：检测会框到整幅光带上，只用手标位置
        return np.array([dflt] * n, np.float32)
    for k, fl in vsd:
        fl = [b for b in fl if (b[2] - b[0]) <= 0.45 * W and (b[3] - b[1]) <= 0.96 * H]   # 框到半幅以上的不是一个人
        fl = sorted(fl, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]
        if len(fl) < 2:
            continue
        fl = sorted(fl, key=lambda b: b[0] + b[2])
        if bs:
            prev = bs[-1]
            if any(abs((fl[j][0] + fl[j][2]) - (prev[j][0] + prev[j][2])) / 2 > 0.25 * W for j in (0, 1)):
                continue
        ks.append(k); bs.append([list(fl[0][:4]), list(fl[1][:4])])
    if not ks:
        return np.array([dflt] * n, np.float32)
    bs = np.array(bs, np.float64)
    out = np.zeros((n, 2, 4), np.float64)
    for j in range(2):
        for c_ in range(4):
            out[:, j, c_] = np.interp(np.arange(n), ks, bs[:, j, c_])
    return gaussian_filter1d(out, 3, axis=0, mode="nearest").astype(np.float32)


def shot_sig(s):
    return [s["code"] if "code" in s else s["yid"], round(s["src_a"], 3), round(s["speed"], 3), s["F_in"], s["F_out"], s["look"],
            s.get("keep_x"), s.get("hold")] + ([s.get("hud"), list(s.get("det") or []), list(s.get("manual") or [])]
                                               if s.get("hud") in ("versus", "link") or "manual" in (s.get("det") or []) else [])


def ana_ok(c, s):
    p = os.path.join(c["ana"], s["id"] + ".pkl")
    if not os.path.exists(p):
        return False
    try:
        return pickle.load(open(p, "rb")).get("sig") == shot_sig(s)
    except Exception:
        return False


def cam_track(frames_g):
    """镜头全局运动：相邻帧 LK 光流 + RANSAC 相似变换（半分辨率算，平移 ×2），累乘成"第 0 帧 → 第 i 帧"。
    镜头内漏切 / 闪白（变换离谱）当作不动。返回 [n, 2, 3]。"""
    C = [np.eye(3)]
    for i in range(1, len(frames_g)):
        M = np.eye(3)
        p0 = cv2.goodFeaturesToTrack(frames_g[i - 1], 300, 0.01, 8)
        if p0 is not None and len(p0) >= 8:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(frames_g[i - 1], frames_g[i], p0, None, winSize=(21, 21), maxLevel=3)
            ok = st.reshape(-1) == 1
            if ok.sum() >= 8:
                A, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if A is not None and inl is not None and inl.sum() >= 6:
                    sc = math.hypot(A[0, 0], A[1, 0])
                    if abs(sc - 1) < 0.12 and abs(A[0, 2]) < 120 and abs(A[1, 2]) < 120:
                        M[:2] = A; M[0, 2] *= 2; M[1, 2] *= 2
        C.append(M @ C[-1])
    return np.array([c_[:2] for c_ in C], np.float32)


def analyze_shot(args):
    key, s = args
    c = prep(key)
    os.makedirs(c["ana"], exist_ok=True)
    src = LiveSource(s["yid"], src_t(s, s["F_in"]), max(src_t(s, s["F_out"] - 1), src_t(s, s["F_in"])) + 0.1)
    acc_tr = L.AccentTrack()
    frames_g, accs, masks, dets, vsd = [], [], [], [], []
    rgb0 = rgb_mid = None
    fdet = None
    if s["look"] in ("Cd", "Cb", "Cs"):
        from anime_det import Det
        fdet = Det("face")
    fgq = []
    for F in range(s["F_in"], s["F_out"]):
        rgb = src.get(src_t(s, F))
        if rgb0 is None:
            rgb0 = rgb.copy()
        if F == (s["F_in"] + s["F_out"]) // 2:
            rgb_mid = rgb.copy()
        frames_g.append(cv2.cvtColor(cv2.resize(rgb, (W // 2, H // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY))
        m = None
        if fdet is not None:
            m = L.seg_mask(rgb)
            if "keep_x" in s:
                m[:, :s["keep_x"]] = 0
            masks.append(cv2.resize((m * 255).astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA))
            if (F - s["F_in"]) % 4 == 0 or F == s["F_out"] - 1:
                dets.append((F - s["F_in"], fdet(rgb, th=0.2)))
        if s["look"] in ("R", "P") and not s.get("nohud"):     # 无 HUD 的镜头不需要前景遮挡图
            mq = L.seg_mask(rgb)
            fgq.append(cv2.resize((mq * 255).astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA))
        if s.get("hud") in ("versus", "link") and (F - s["F_in"]) % 4 == 0:   # 对决牌：逐 4 帧人物检测
            from anime_det import Det
            vsd.append((F - s["F_in"], Det("person")(rgb, th=0.25)))
        accs.append(acc_tr(rgb, m))
    ana = dict(id=s["id"], F_in=s["F_in"], accs=accs, sig=shot_sig(s), cam=cam_track(frames_g))
    if fgq:
        np.save(os.path.join(c["ana"], s["id"] + "_fgq.npy"), np.array(fgq))
    if s["look"] in ("M", "B", "A", "G", "T"):
        from anime_det import Det
        ref = rgb_mid if (s["look"] == "G" and rgb_mid is not None) else rgb0      # 灵基卡：镜头中间那帧找脸（开头可能是远景）
        fb = Det("face")(ref, th=0.25) or Det("head")(ref, th=0.3)
        ana["face_rel"] = (((fb[0][0] + fb[0][2]) / 2) / W, (fb[0][1] * 0.6 + fb[0][3] * 0.4) / H) if fb else None
        ana["how"] = ""
        pickle.dump(ana, open(os.path.join(c["ana"], s["id"] + ".pkl"), "wb"))
        return s["id"], len(accs), "M/B"
    if s["look"] in ("R", "P"):
        box0, how = pick_box(rgb0, s["det"], s["manual"])
        ana["boxes"] = box_track(frames_g, box0); ana["how"] = how
        if s.get("hud") in ("versus", "link"):
            ana["vs_boxes"] = vs_track(s, vsd, len(frames_g))
    else:
        faces = dets[0][1]
        if "keep_x" in s:
            faces = [b for b in faces if (b[0] + b[2]) / 2 >= s["keep_x"]]
        ana["faces0"] = faces
        n = len(frames_g)
        fcs = np.zeros((n, len(faces), 2), np.float32)
        for j, b in enumerate(faces):                      # 每张初始脸：逐检测帧找最近框（≤250 px），缺的插值，再平滑
            prev = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            ks, xs = [], []
            for k, fl in dets:
                cs = [np.array([(q[0] + q[2]) / 2, (q[1] + q[3]) / 2]) for q in fl]
                if cs:
                    dd = [np.linalg.norm(cc - prev) for cc in cs]
                    if min(dd) < 250:
                        prev = cs[int(np.argmin(dd))]; ks.append(k); xs.append(prev)
            if not ks:
                ks, xs = [0], [prev]
            xs = np.array(xs)
            for ax in (0, 1):
                fcs[:, j, ax] = np.interp(np.arange(n), ks, xs[:, ax])
        if len(faces):
            fcs = gaussian_filter1d(fcs, 2.5, axis=0, mode="nearest")
        ana["face_c"] = fcs
        m0 = cv2.resize(masks[0].astype(np.float32) / 255, (W, H))
        if s["look"] == "Cd" and faces:                    # 只要主角：脸所在的连通域
            n_, lab = cv2.connectedComponents((m0 > 0.5).astype(np.uint8))
            keep = lab[int(np.clip(fcs[0, 0, 1], 0, H - 1)), int(np.clip(fcs[0, 0, 0], 0, W - 1))]
            ana["keep_xy"] = tuple(fcs[0, 0])
            if keep:
                m0 = m0 * (lab == keep)
            nodes0 = np.array(L.nodes_from_mask(m0, faces[:1], n=7), np.float32)
            ana["nodes"] = nodes0[None] + (fcs[:, :1] - fcs[:1, :1])     # 节点跟着脸刚性平移
        else:
            ana["nodes"] = np.zeros((n, 0, 2), np.float32)
        ana["mask0"] = m0
        masks, ana["mask_fixed"] = repair_masks(masks, ana["cam"])
        np.save(os.path.join(c["ana"], s["id"] + "_mask.npy"), np.array(masks))
    pickle.dump(ana, open(os.path.join(c["ana"], s["id"] + ".pkl"), "wb"))
    return s["id"], len(accs), ana.get("how", "")


# ---------------------------------------------------------------- render
class Ctx:
    def __init__(self, c):
        self.c = c
        self.ana = {s["id"]: pickle.load(open(os.path.join(c["ana"], s["id"] + ".pkl"), "rb")) for s in c["shots"]}
        self.masks = {s["id"]: np.load(os.path.join(c["ana"], s["id"] + "_mask.npy"), mmap_mode="r")
                      for s in c["shots"] if s["look"] in ("Cd", "Cb", "Cs")}
        self.src = {}
        if c.get("hud_v", 3) >= 5:                         # v6：HUD v5 模板（精密 UI + 编排动效 + 角色 / 画面配色），只写核对过的信息
            import hud_v5 as H5
            self.hud = {s["id"]: H5.HudV5(i, s.get("hud", "lock"), H5.info_for(s["who"], s.get("title", ""), s.get("tags", [])),
                                          H5.WORK.get(s.get("group", ""), "")) for i, s in enumerate(c["shots"]) if s["look"] in ("R", "P")}
        else:
            self.hud = {s["id"]: HudV3(i, s["title"], s["tags"], CALL) for i, s in enumerate(c["shots"]) if s["look"] in ("R", "P")}
        for s_ in c["shots"]:                              # v10：break 里的 HUD 知道外面有 21:9 遮幅（取景器不再自己画遮幅）
            if s_.get("cine") and s_["id"] in self.hud:
                self.hud[s_["id"]].cine = True
        self.comic = {}
        self.fgq = {s["id"]: np.load(os.path.join(c["ana"], s["id"] + "_fgq.npy"), mmap_mode="r")
                    for s in c["shots"] if s["look"] in ("R", "P") and os.path.exists(os.path.join(c["ana"], s["id"] + "_fgq.npy"))}
        self.spring, self.manga, self.riso, self.snaps = {}, {}, {}, {}
        self.cur_shot = None
        self.cache_glyph = {}
        self.rms = L.rms60()

    def grab(self, s, F):
        """单帧取源（漫画分镜定格格子用），带缓存。"""
        k = (s["id"], F)
        if k not in self.snaps:
            from cards import src_path
            ts = src_t(s, F)
            p = subprocess.run([SF.FF, "-v", "error", "-ss", "%.4f" % ts, "-i", src_path(s["yid"]), "-frames:v", "1",
                                "-vf", "scale=%d:%d:flags=lanczos" % (W, H), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
            img = np.frombuffer(p.stdout, np.uint8)[:W * H * 3].reshape(H, W, 3).copy()
            if s.get("crop"):
                kk = s["crop"]
                ty = H * (1 - kk) if s.get("crop_anchor") == "bottom" else 0.0
                img = cv2.warpAffine(img, np.float32([[kk, 0, W / 2 * (1 - kk)], [0, kk, ty]]), (W, H), flags=cv2.INTER_LINEAR)
            self.snaps[k] = img
        return self.snaps[k]

    def frame(self, s, F):
        key = s["id"]
        ts = src_t(s, F)
        for k_ in [k_ for k_, v in self.src.items() if F >= v[2] or F < v[3] - 1]:   # 已经播完的镜头：关掉它的 ffmpeg 解码进程
            self.src[k_][0].close(); del self.src[k_]
        if key not in self.src or self.src[key][1] > ts:
            if key in self.src:
                self.src[key][0].close()
            self.src[key] = (LiveSource(s["yid"], ts, max(ts, src_t(s, s["F_out"] - 1)) + 0.1), ts, s["F_out"], s["F_in"])
        img = self.src[key][0].get(ts)
        if s.get("crop"):                                  # 卡尔纳 vs 齐格飞水印：绕顶边中点放大，推出画面
            k = s["crop"]
            ty = H * (1 - k) if s.get("crop_anchor") == "bottom" else 0.0          # 水印在上就锚底边，在下就锚顶边
            img = cv2.warpAffine(img, np.float32([[k, 0, W / 2 * (1 - k)], [0, k, ty]]), (W, H), flags=cv2.INTER_LINEAR)
        return img


def hud_on(ctx, s, base, F, f, acc, i, tear=0.0, src=None):
    a = ctx.ana[s["id"]]
    bk, bf = beat_state(F)
    n = s["Fb"] - s["Fa"]
    i0 = int(np.clip(s["Fa"] - a["F_in"], 0, len(a["boxes"]) - 1))
    if s["id"] not in ctx.spring:                          # 取景框：阻尼弹簧追踪（入场从画外 ~320R 飞来锁定）
        bx = np.asarray(a["boxes"], np.float64)
        cen = np.column_stack([(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2])
        rng = np.random.default_rng(sum(map(ord, s["id"])))
        ang = rng.uniform(0, 2 * math.pi)
        sp = spring_track(cen[i0:], start_off=(320 * R * math.cos(ang), 220 * R * math.sin(ang)))
        ctx.spring[s["id"]] = np.vstack([np.repeat(sp[:1], i0, 0), sp]) if i0 else sp
    c_ = ctx.spring[s["id"]][min(i, len(ctx.spring[s["id"]]) - 1)]
    bx = a["boxes"][i]; hw, hh = (bx[2] - bx[0]) / 2, (bx[3] - bx[1]) / 2
    box = (c_[0] - hw, c_[1] - hh, c_[0] + hw, c_[1] + hh)
    active = max(54, int(0.62 * n))                        # 用户："覆盖时间太长"——HUD 只占镜头前 ~60%（至少 0.9 s），然后收掉
    life_in = 1.0 if (s.get("burst") and f >= 0) else cl(f / 14.0)
    life_out = cl((f - active) / 12.0) if active < n - 6 else 0.0          # v5 退场是分层反序收回，给 12 帧
    cam = a.get("cam")
    cam_rel, vel = None, (0.0, 0.0)
    if cam is not None and len(cam):
        C = np.eye(3); C[:2] = cam[min(i, len(cam) - 1)]
        C0 = np.eye(3); C0[:2] = cam[i0]
        cam_rel = (C @ np.linalg.inv(C0))[:2]
        if i > 0:
            vel = (float(cam[i][0, 2] - cam[i - 1][0, 2]), float(cam[i][1, 2] - cam[i - 1][1, 2]))
    fg = None
    if s["id"] in ctx.fgq:                                 # 前景人物遮挡后景 HUD（时间上取 3 帧平均，边缘稳）
        Q = ctx.fgq[s["id"]]; j = min(i, len(Q) - 1)
        q = Q[max(0, j - 1):j + 2].astype(np.float32).mean(0) / 255
        fg = cv2.resize(cv2.GaussianBlur(q, (0, 0), 1.2), (W, H), interpolation=cv2.INTER_LINEAR)
    slot = s["b"] - s["a"]
    ns = 0 if (s.get("bars") or slot < 0.9) else 1
    g = glitch_amp(ctx.c, F) * 0.9                         # HUD 层 glitch：全局包络 / 爆点炸出后 0.5 s / 高潮段每拍脉冲 / 版画镜头底噪
    if s.get("burst") and 0 <= f < 30:
        g = max(g, 0.6 * (1 - f / 30.0))
    if s.get("hot") and bf < 4:
        g = max(g, 0.45 * (1 - bf / 4.0))
    if s["look"] == "P":
        g = max(g, 0.15)
    vs = None
    if s.get("hud") in ("versus", "link"):
        vb = a.get("vs_boxes")
        vs = dict(s.get("vs") or {}, boxes=(vb[min(i, len(vb) - 1)].tolist() if vb is not None else None))
    path = ctx.spring[s["id"]][max(i0, 0):i + 1] if s.get("hud") == "track" else None
    return ctx.hud[s["id"]].render3(base, max(f, 0), F / FPS, bk, bf, box, acc, ctx.rms, src=src, life_in=life_in, life_out=life_out,
                                    cam=cam_rel, cam_vel=vel, fg=fg, n_samples=ns, burst=s.get("burst", False) and f >= 0,
                                    hud_glitch=g, gseed=F // 2, kind=a["how"], path=path, vs=vs)


def _rel(cam, i, j):
    """把第 j 帧的半分辨率图对齐到第 i 帧：C_i · C_j⁻¹（cam 是全分辨率"第 0 帧 → 第 i 帧"，平移减半）。"""
    Ci = np.vstack([cam[i], [0, 0, 1]]); Cj = np.vstack([cam[j], [0, 0, 1]])
    M = (Ci @ np.linalg.inv(Cj))[:2].astype(np.float32)
    M[:, 2] *= 0.5
    return M


def repair_masks(masks, cam, win=24, reach=30):
    """isnetis 偶尔整帧漏抠（E12 马修 167.4–167.6 s 连续 9 帧全 0，CPU / 显卡都一样，用户第 6 条）：
    覆盖率掉到前 / 后 win 帧中位数（取大的）40% 以下、且前后 reach 帧内都有好帧的（真漏抠，不是人物出入画），
    用前后最近的好帧按镜头运动 warp 过来、按距离混合补上。返回 (masks, 补了几帧)。"""
    masks = np.asarray(masks)
    n = len(masks)
    cov = (masks > 128).mean((1, 2))
    bad = np.zeros(n, bool)
    for i in range(n):
        l_, r_ = cov[max(0, i - win):i], cov[i + 1:i + 1 + win]
        med = max(float(np.median(l_)) if l_.size else 0.0, float(np.median(r_)) if r_.size else 0.0)
        bad[i] = med > 0.01 and cov[i] < 0.4 * med
    good = np.where(~bad)[0]
    out = masks.copy(); fixed = 0
    for i in np.where(bad)[0]:
        lo, hi = good[good < i], good[good > i]
        if not (lo.size and hi.size and i - lo[-1] <= reach and hi[0] - i <= reach):
            continue
        acc = np.zeros(masks.shape[1:], np.float32); ws = 0.0
        for j in (lo[-1], hi[0]):
            w = 1.0 / abs(i - j)
            acc += w * cv2.warpAffine(masks[j].astype(np.float32), _rel(cam, i, j), (masks.shape[2], masks.shape[1]))
            ws += w
        out[i] = np.clip(acc / ws, 0, 255).astype(np.uint8); fixed += 1
    return out, fixed


def mask_smooth(ctx, s, i, rad=2):
    """C 系抠像：±2 帧加权平均后低阈值（发丝不再一帧有一帧无，用户第 6 条），再膨胀 + 软边。
    邻帧先试"按镜头运动对齐"和"不对齐"两种，取和当前帧重合度（IoU）高的；IoU 低于 0.8 的邻帧不参与
    （人物自己在动 / 视差大时平均会拖出残影，E12 马修 167.0 s）。"""
    M = ctx.masks[s["id"]]; j = min(i, len(M) - 1)
    cam = ctx.ana[s["id"]].get("cam")
    cur = M[j].astype(np.float32)
    cs = cur[::4, ::4] > 128

    def iou(x):
        xs = x[::4, ::4] > 128
        u = (xs | cs).sum()
        return 1.0 if u == 0 else float((xs & cs).sum()) / u
    acc = cur.copy(); ws = 1.0
    for k in range(max(0, j - rad), min(len(M), j + rad + 1)):
        if k == j:
            continue
        cands = [M[k].astype(np.float32)]
        if cam is not None and len(cam) > max(j, k):
            cands.append(cv2.warpAffine(cands[0], _rel(cam, j, k), (M.shape[2], M.shape[1])))
        sc = [iou(x) for x in cands]
        b_ = int(np.argmax(sc))
        w = float(np.clip((sc[b_] - 0.80) / 0.12, 0, 1))
        if w > 0:
            acc += w * cands[b_]; ws += w
    m = acc / ws / 255
    m = np.clip((m - 0.25) / 0.30, 0, 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)


def styled(ctx, s, F):
    """单镜头风格化画面（不含歌词、不含全局 glitch）。"""
    rgb = ctx.frame(s, F)
    a = ctx.ana[s["id"]]
    i = int(np.clip(F - a["F_in"], 0, len(a["accs"]) - 1))
    acc = PAL.shot_acc(s.get("who", ""), a["accs"][i])    # 人物配色（用户："少用橙色配色，你要根据人物配色来选择风格化配色"）
    f = F - s["Fa"]
    if s.get("impact") and 0 <= f < s["impact"]:          # 爆点冲击帧：版画 + 撕裂 + 金色放射细线（glitch 由全局包络给满）
        img = L.wave_tear(L.poster_h(rgb, F, acc, seed=7), 0.5, tint=acc["sec"])
        bx = a["boxes"][i]; cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        rng = np.random.default_rng(F)
        for _ in range(48):
            ang = rng.uniform(0, 2 * math.pi); r0 = H * rng.uniform(0.16, 0.30)
            p0 = (int((cx + r0 * math.cos(ang)) * 16), int((cy + r0 * math.sin(ang)) * 16))
            p1 = (int((cx + 2 * W * math.cos(ang)) * 16), int((cy + 2 * W * math.sin(ang)) * 16))
            cv2.line(img, p0, p1, L.GOLD_H, max(1, int(rng.uniform(1, 3) * R)), cv2.LINE_AA, 4)
        return img, acc
    if s["look"] == "T":                                    # v9 成片：歌名 / 标题卡（召唤阵做底）
        return TT.title_card(L.r_base(rgb, F), max(f, 0), s["Fb"] - s["Fa"], acc, s["title_spec"]), acc
    if s["look"] == "A":                                    # 美漫（用户 v3 后第 3 条）：一拍二（12 fps 换画）
        k2 = (s["id"], F // 2)
        if k2 not in ctx.comic:
            ctx.comic = {k_: v for k_, v in ctx.comic.items() if k_[0] == s["id"] and k_[1] >= F // 2 - 1}
            ctx.comic[k2] = V4.comic_us(rgb, max(f, 0), acc, sum(map(ord, s["id"])), caption=s.get("caption"),
                                        sfx=s.get("sfx") if 0 <= f < 48 else None, key=s["id"], tag=s.get("cap_tag"))
        return ctx.comic[k2], acc
    if s["look"] == "G":                                    # Fate 灵基卡（用户 v3 后第 4 条）
        inf = H4.info_for(s["who"], s.get("title", ""), s.get("tags", []))
        return V4.saint_graph(rgb, max(f, 0), acc, inf, a.get("face_rel"), sum(map(ord, s["id"]))), acc
    if s["look"] == "Cs":                                   # 片尾马修：不抠头，脸周围一大块柔边椭圆保留原色（用户："不用完全将遮罩缩小在头部范围"）
        if s["id"] not in ctx.__dict__.setdefault("cs_track", {}):     # 人物中心轨迹：抠像质心（缺的插值）再时间平滑 σ=10 帧
            M_ = ctx.masks[s["id"]]
            cs = np.full((len(M_), 2), np.nan, np.float32)
            for k_ in range(len(M_)):
                q_ = np.asarray(M_[k_][::4, ::4]) > 128
                if q_.mean() > 0.02:
                    yy_, xx_ = np.nonzero(q_)
                    cs[k_] = (xx_.mean() * 8, np.percentile(yy_, 20) * 8)
            ok_ = ~np.isnan(cs[:, 0])
            if ok_.any():
                for ax_ in (0, 1):
                    cs[:, ax_] = np.interp(np.arange(len(cs)), np.nonzero(ok_)[0], cs[ok_, ax_])
                cs = gaussian_filter1d(cs, 10, axis=0, mode="nearest")
            else:
                cs[:] = (W * 0.5, H * 0.42)
            ctx.cs_track[s["id"]] = cs
        fc = ctx.cs_track[s["id"]][min(i, len(ctx.cs_track[s["id"]]) - 1)]
        if "soft" not in ctx.__dict__:
            yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
            ctx.soft = (yy * 4, xx * 4)
        yy, xx = ctx.soft
        d = np.hypot((xx - fc[0]) / (0.36 * W), (yy - fc[1] - 0.10 * H) / (0.62 * H))
        m = cv2.resize(np.clip((1.0 - d) / 0.45, 0, 1) ** 1.3, (W, H), interpolation=cv2.INTER_LINEAR)
        base, lay = L.c_dark(rgb, m, max(f, 0), [], [], acc=acc)
        return L.compose(base, lay), acc
    if s["look"] == "M":                                    # 漫画分镜（用户第 2 条）
        mp = ctx.manga.get(s["id"])
        if mp is None:
            cap = s.get("caption")
            if s.get("caption_lyr") is not None:            # 歌词行进竖排框：文字从用户的歌词包取，EDL 里只有行号
                cap = LYR[s["caption_lyr"]]["text"].replace(" ", "").replace("　", "")
            mp = MG.MangaPage(s["manga"], seed=sum(map(ord, s["id"])), face=a.get("face_rel"), tone=s.get("tone", "gray"), acc=acc,
                              caption=cap, num=s.get("num", 7), tag=s.get("tag", ""), smooth=bool(s.get("cine")))
            mp.duo = s.get("duo")
            ctx.manga[s["id"]] = mp
        mp.acc = acc
        snap = (lambda fk: ctx.grab(s, s["Fa"] + fk)) if s["manga"] == "story" else None
        return mp.frame(max(f, 0), rgb, snap=snap), acc
    if s["look"] == "B":                                    # riso 角色海报
        rp = ctx.riso.get(s["id"])
        if rp is None:
            po = s["poster"]
            rp = MG.RisoPoster(po["big"], po["sub"], po["jp"], po.get("num", "01"), smooth=bool(s.get("cine")))
            ctx.riso[s["id"]] = rp
        return rp.frame(max(f, 0), rgb), acc
    if s["look"] in ("R", "P"):
        tear = 0.0
        if s["look"] == "R":
            base = rgb.copy() if s.get("cine") else L.r_base(rgb, F)   # v10：break 镜头不要 r_base 的冷色网点（电影质感层在整帧后统一做）
        else:
            base = L.poster_h(rgb, F, acc, seed=sum(map(ord, s["id"])), mode="tension" if s.get("tension") else "full")
            if s.get("boil"):
                base = L.boil(base, F)
        for tb in s.get("tear", []):                      # 重拍波形撕裂：3 帧出、8 帧回
            d = F - fr(tb) - (s.get("impact", 0))
            if 0 <= d < 11:
                k = 0.5 * d / 3 if d < 3 else 0.5 + 0.5 * (d - 3) / 8
                base = L.wave_tear(base, k, tint=acc["sec"]); tear = 1.0
        img = base if s.get("nohud") else hud_on(ctx, s, base, F, f, acc, i, tear, src=rgb)   # v10：块内连续镜头只在开场 / 宝具镜头上 HUD
        if s["look"] == "P" and (s.get("cropbox") or s.get("growth")):
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            if s.get("growth"):                            # 角落锈块蔓延（causasui 像素生长，≤10%）
                seeds = [(0, H, 520), (W, 0, 520), (W, H, 380)]
                L.pixel_growth(lay, np.ones((H, W), np.float32), np.zeros((H, W), np.uint8), 0.95 * eo(f / 30.0), seeds, col=acc["main"])
            if s.get("cropbox") and f >= 3:                # 裁切框拍上（causasui）：取目标框的彩色特写
                bx = a["boxes"][i]
                cxb = (bx[0] + bx[2]) / 2
                w_ = min(bx[2] - bx[0], W * 0.22); h_ = w_ * 0.58
                rect = (cxb - w_ / 2, bx[1] + (bx[3] - bx[1]) * 0.30, w_, h_)
                dst = (W * 0.62, H * 0.18) if cxb < W / 2 else (W * 0.05, H * 0.36)
                L.crop_box(lay, rgb, rect, dst, (int(500 * R), int(290 * R)), s["cropbox"], min(1.0, (f - 3) / 5.0), acc=acc)
            img = L.compose(img, lay)
        if s.get("push"):                                  # 换气段缓推（ease-in，镜头不抖）
            k = (F - s["Fa"]) / float(s["Fb"] - s["Fa"])
            img = T.slam(img, 1.0, s0=1.0) if k <= 0 else cv2.warpAffine(
                img, np.float32([[1 + s["push"] * k * k, 0, -W / 2 * s["push"] * k * k], [0, 1 + s["push"] * k * k, -H / 2 * s["push"] * k * k]]),
                (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if s.get("slam") and s.get("impact", 0) <= f < s.get("impact", 0) + s["slam"]:
            img = T.slam(img, (f - s.get("impact", 0)) / float(s["slam"]), s0=1.22)
        return img, acc
    m = mask_smooth(ctx, s, i)
    if s["look"] == "Cb" and (m > 0.5).mean() < 0.04:      # 抠像几乎为空（暗场 / 特写失败）：整帧都进档案色调，不留空白纸（用户第 9 条 0:17）
        m = np.ones((H, W), np.float32)
    if s["look"] == "Cd":
        if "keep_xy" in a:                                 # 跟着脸走的那个连通域
            n_, lab = cv2.connectedComponents((m > 0.5).astype(np.uint8))
            fx, fy = a["face_c"][i][0] if a["face_c"].shape[1] else a["keep_xy"]
            keep = lab[int(np.clip(fy, 0, H - 1)), int(np.clip(fx, 0, W - 1))]
            if keep:
                m = m * (cv2.dilate((lab == keep).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
        nodes = [tuple(map(int, p)) for p in a["nodes"][i]]
        base, lay = L.c_dark(rgb, m, max(f, 0), nodes, s["labels"][:len(nodes)], acc=acc)
        return L.compose(base, lay), acc
    base = L.c_bright(rgb, m, acc)                         # Cb 档案亮态
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    seeds = [(W * 0.30, H * 0.85, 360), (W * 0.62, H * 0.95, 260), (W * 0.40, H * 0.10, 200)]
    L.pixel_growth(lay, a["mask0"], L.face_mask(a["faces0"], 8), 0.85 * eo((f - 6) / 50.0), seeds, col=acc["main"])
    faces = sorted(zip(a["faces0"], range(len(a["faces0"]))), key=lambda z: z[0][0])
    dsts = [((W * 0.04, H * 0.10), (int(560 * R), int(330 * R))), ((W * 0.04, H * 0.52), (int(560 * R), int(330 * R)))]
    for (b, idx), title, (dst, size), tb in zip(faces, s["titles"], dsts, s["box_beats"]):
        k = (F - fr(tb)) / 5.0
        if k <= 0:
            continue
        cx, cy = a["face_c"][i][idx]
        w_ = b[2] - b[0]; h_ = w_ * size[1] / size[0]
        x0_, y0_ = int(np.clip(cx - w_ / 2, 0, W - 2)), int(np.clip(cy - (b[3] - b[1]) / 2, 0, H - 2))
        patch_ = rgb[y0_:y0_ + max(2, int(h_)), x0_:x0_ + max(2, int(w_))]
        if patch_.size == 0 or float(patch_.std()) < 9.0 or float(patch_.mean()) < 16.0:
            continue
        L.crop_box(lay, rgb, (cx - w_ / 2, cy - (b[3] - b[1]) / 2 + (b[3] - b[1]) * 0.22, w_, h_), dst, size, title, min(k, 1.0), acc=acc)
    d = ImageDraw.Draw(lay)
    tw = int(W * 0.30) * eo((f - 4) / 8.0)
    if tw > 2:
        x1 = W - 60 * R; x0 = x1 - tw
        d.rectangle([x0, 60 * R, x1, 104 * R], fill=acc["main"] + (255,))
        if tw > W * 0.25:
            arch = {"HFM": "HEAVEN'S FEEL", "FSF": "STRANGE FAKE", "FGO": "CHALDEA", "UBW": "UNLIMITED BLADE WORKS",
                    "APO": "APOCRYPHA", "HFG": "HEAVEN'S FEEL", "OTH": "RECORD"}.get(s.get("group"), "APOCRYPHA")   # 档案名跟片源系列走
            d.text((x0 + 12 * R, 66 * R), "ARCHIVE // " + arch, font=font(L.F_WIDE, 22 * R), fill=acc["txt"] + (255,))
    return L.compose(base, lay), acc


# ---- 歌词
def glyphs(ctx, text, size, fill, jp, en, en_scale=1.0):
    key = (text, size, fill, jp, en, en_scale)
    if key in ctx.cache_glyph:
        return ctx.cache_glyph[key]
    fj = SF.vfont(jp, size, "SemiBold") if jp == L.F_JPS else font(jp, size)
    fe = font(en, size * en_scale)
    asc = max(fj.getmetrics()[0], fe.getmetrics()[0]); desc = max(fj.getmetrics()[1], fe.getmetrics()[1])
    out, x = [], 0.0
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    for ch in text:
        f_ = fj if L.is_cjk(ch) else fe
        adv = probe.textlength(ch, font=f_)
        im = Image.new("RGBA", (int(adv + size * 0.3) + 2, asc + desc + 4), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, asc), ch, font=f_, fill=fill, anchor="ls")
        out.append((im, x, ch)); x += adv
    ctx.cache_glyph[key] = (out, x, asc + desc + 4)
    return ctx.cache_glyph[key]


def put(lay, im, x, y, alpha=1.0, scale=1.0, sx=1.0):
    if alpha <= 0.01:
        return
    if abs(scale - 1) > 0.01 or abs(sx - 1) > 0.004:
        w_, h_ = max(1, int(im.width * scale * sx)), max(1, int(im.height * scale))
        x -= (w_ - im.width) / 2; y -= (h_ - im.height) / 2
        im = im.resize((w_, h_), Image.BILINEAR)
    if alpha < 0.999:
        im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    lay.alpha_composite(im, (int(x), int(y)))


SCR = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789#%&/<>=+"


def word_font(word, size):
    if L.is_cjk(word[0]):
        return SF.vfont(L.F_JPS, size, "Black")
    return SF.vfont(L.F_EN, size, "Bold SemiCondensed")


def word_size(word, last):
    n = len(word)
    if L.is_cjk(word[0]):
        sz = 300 if n <= 2 else 260 if n <= 3 else 220 if n <= 4 else 180
    else:
        sz = 330 if n <= 4 else 260 if n <= 7 else 200
    return int(sz * R * (1.12 if last else 1.0))


def word_img(ctx, word, fnt_size, fill=None, stroke=None, sw=0):
    key = ("w", word, fnt_size, fill, stroke, sw)
    if key in ctx.cache_glyph:
        return ctx.cache_glyph[key]
    fnt = word_font(word, fnt_size)
    d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    tw = d0.textlength(word, font=fnt); asc, desc = fnt.getmetrics()
    pad = int(30 * R)
    im = Image.new("RGBA", (int(tw) + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
    dd = ImageDraw.Draw(im)
    if stroke is not None:
        dd.text((pad, pad), word, font=fnt, fill=(0, 0, 0, 0), stroke_width=sw, stroke_fill=stroke)
        dd.text((pad, pad), word, font=fnt, fill=(0, 0, 0, 0))
    else:
        dd.text((pad, pad), word, font=fnt, fill=fill)
    if len(ctx.cache_glyph) > 400:
        ctx.cache_glyph.clear()
    ctx.cache_glyph[key] = (im, tw, asc + desc, pad)
    return ctx.cache_glyph[key]


def slices(im, amt, seed, n=7):
    """把词图横切成 n 条，奇偶反向错开 amt 像素（glitch / 碎裂退场用）。"""
    if abs(amt) < 1:
        return im
    rng = np.random.default_rng(seed)
    m = int(abs(amt)) + 1
    out = Image.new("RGBA", (im.width + 2 * m, im.height), (0, 0, 0, 0))
    hs = np.sort(rng.uniform(0, im.height, n - 1)).astype(int)
    edges = [0] + list(hs) + [im.height]
    for j in range(n):
        y0, y1 = edges[j], edges[j + 1]
        if y1 <= y0:
            continue
        dx = int(amt * (1 if j % 2 == 0 else -1) * rng.uniform(0.4, 1.0))
        out.alpha_composite(im.crop((0, y0, im.width, y1)), (m + dx, y0))
    return out.crop((m, 0, m + im.width, im.height))


def burst_layer(ctx, sub, i, F, acc_now, hot, wall):
    """逐词炸出（一次一个词）。hot = 高潮：乱码解码入场、回声拖尾、每拍脉冲 + 色散 + 切片、荧光条、碎裂退场；
    wall = 词墙（峰值词：整屏空心描边行反向滚动）。日文按短语切块，用明朝 Black。"""
    words, ts = burst_words(i), burst_times(i)[1]
    j = max([k for k, t_ in enumerate(ts) if fr(t_) <= F] or [0])
    fw = F - fr(ts[j])
    word = words[j]
    cs_ = getattr(ctx, "cur_shot", None)
    if cs_ is not None and cs_.get("look") == "M" and cs_.get("caption_lyr") == i and L.is_cjk(word[0]):
        return                                             # 这句日文已进漫画竖排说明框
    if cs_ is not None and cs_.get("look") == "B":
        return                                             # 角色海报上不叠歌词
    last = j == len(words) - 1
    cjk = L.is_cjk(word[0])
    size = word_size(word, last)
    im0, tw, th_, pad = word_img(ctx, word, size, fill=(244, 242, 236, 255))
    if tw > W * 0.9:                                       # 太宽就缩字号
        size = int(size * W * 0.9 / tw)
        im0, tw, th_, pad = word_img(ctx, word, size, fill=(244, 242, 236, 255))
    box = ctx.cur_box                                      # 避脸：人物框只算上 35%（头），三档高度里挑与脸重叠最少的
    if box is not None and box[3] - box[1] > 0.6 * H:
        box = (box[0], box[1], box[2], box[1] + 0.35 * (box[3] - box[1]))
    slots = [0.30, 0.68, 0.42, 0.58]
    xc = W * (0.5 if last else slots[j % 4])
    hud_ = cs_.get("hud") if cs_ is not None and cs_.get("look") in ("R", "P") else None
    if hud_ in ("dossier", "codex", "blueprint") and box is not None:   # 档案面板 / 竖排 / 制图标注在目标的远侧：歌词放另一半
        panel_r = (box[0] + box[2]) / 2 < W / 2
        xc = W * (0.30 if panel_r else 0.70) if not last else W * (0.36 if panel_r else 0.64)
    xc = float(np.clip(xc, tw / 2 + 60 * R, W - tw / 2 - 60 * R))

    def ov(yc_):
        if box is None:
            return 0.0
        ix = max(0.0, min(xc + tw / 2, box[2]) - max(xc - tw / 2, box[0]))
        iy = max(0.0, min(yc_ + th_ / 2, box[3]) - max(yc_ - th_ / 2, box[1]))
        return ix * iy
    pref = [0.70, 0.32, 0.52] if j % 2 == 0 else [0.32, 0.70, 0.52]
    if hud_ == "np" and box is not None:                   # 宝具横幅在目标的另一边（上 / 下）：歌词不压横幅
        band_low = (box[1] + 0.4 * (box[3] - box[1])) < H * 0.55
        pref = [p_ for p_ in pref if (p_ < 0.6 if band_low else p_ > 0.45)]
    yc = H * min(pref, key=lambda p_: (ov(H * p_), pref.index(p_)))
    yc = float(np.clip(yc, th_ / 2 + 40 * R, H - th_ / 2 - 90 * R))
    x, y = xc - tw / 2, yc - th_ / 2
    nxt = fr(ts[j + 1]) if j + 1 < len(ts) else None
    seed = i * 100 + j
    dur_w = ((nxt if nxt is not None else fr(lyr_end(i))) - fr(ts[j])) / FPS
    slow = (not hot) and dur_w >= 0.45                     # 拖长音的词：慢入场（9 帧）+ 唱的时候字距慢慢拉开
    if wall and (wall is True or (wall == "last" and last) or (isinstance(wall, list) and j in wall)):
        wim = word_img(ctx, word, int(size * 0.55), stroke=tuple(acc_now["hi"]) + (70,), sw=max(1, int(1.5 * R)))[0]
        rowh = int(wim.height * 0.78)
        stepx = wim.width + int(40 * R)
        for r_ in range(-1, H // rowh + 2):                # 词墙：奇偶行反向滚动
            off = int(fw * 7 * R * (1 if r_ % 2 == 0 else -1)) % stepx - stepx
            cx_ = off - (stepx // 2 if r_ % 2 else 0)
            while cx_ < W:
                sub.alpha_composite(wim, (int(cx_), int(r_ * rowh - wim.height * 0.1))) if cx_ + wim.width > 0 else None
                cx_ += stepx
    if hot:                                                # 强调色荧光条
        bw_ = tw * T.expo_out(fw / 6.0)
        ImageDraw.Draw(sub).rectangle([x, y + th_ * 0.55, x + bw_, y + th_ * 0.92], fill=tuple(acc_now["main"]) + (255,))
    k = T.expo_out(fw / 4.0)
    if slow and fw < 9:                                    # 慢入场：描边从 1.18 倍缓缓落位、渐显，色散很轻
        kk = T.expo_out(fw / 9.0)
        sp = int(6 * R * (1 - kk))
        for col, dx in ((acc_now["main"], -sp), (acc_now["sec"], sp)):
            put(sub, word_img(ctx, word, size, stroke=tuple(col) + (200,), sw=int(3 * R))[0], x - pad + dx, y - pad, alpha=kk, scale=1.18 - 0.18 * kk)
        put(sub, word_img(ctx, word, size, fill=(244, 242, 236, 255))[0], x - pad, y - pad, alpha=kk ** 2, scale=1.18 - 0.18 * kk)
    elif fw < 3:                                           # 入场：空心描边 + 双色色散（亮度安全）；高潮时先 2 帧乱码解码
        sc = 1.5 - 0.5 * k
        txt = word
        if hot and not cjk and fw < 2:
            rng = np.random.default_rng(seed * 7 + fw)
            txt = "".join(SCR[int(rng.integers(len(SCR)))] if ch != " " else " " for ch in word)
        sp = int(14 * R * (1 - k))
        for col, dx in ((acc_now["main"], -sp), (acc_now["sec"], sp)):
            put(sub, word_img(ctx, txt, size, stroke=tuple(col) + (230,), sw=int(3 * R))[0], x - pad + dx, y - pad, scale=sc)
        put(sub, word_img(ctx, txt, size, stroke=(246, 244, 240, 255), sw=int(3 * R))[0], x - pad, y - pad, scale=sc)
    else:
        sc = 1.0 - 0.03 * min(1.0, (fw - 3) / 60.0)
        pulse = 0.0
        for tp in ctx.c.get("punch", []):                  # 爆点再砸一次：1.3→1
            dp = F - fr(tp)
            if 0 <= dp < 8:
                kp = T.expo_out(dp / 6.0); sc = 1.3 - 0.3 * kp; pulse = max(pulse, 1 - kp)
        if hot:                                            # 每拍一次脉冲（1.04→1 + 色散 + 切片）
            bk, bf = beat_state(F)
            if bf < 4 and fw > 4:
                pulse = max(pulse, 1 - bf / 4.0); sc *= 1 + 0.04 * (1 - bf / 4.0)
        if hot and fw < 11:                                # 回声拖尾：3 层空心描边由大到小收回
            ke = (fw - 3) / 8.0
            for m_ in (3, 2, 1):
                put(sub, word_img(ctx, word, size, stroke=tuple(acc_now["hi"]) + (int(150 / m_),), sw=int(2 * R))[0],
                    x - pad, y - pad, scale=sc * (1 + 0.07 * m_ * (1 - ke)))
        im_sh = word_img(ctx, word, size, fill=tuple(acc_now["main"]) + (255,))[0]
        im_w = im0
        g_amt = 18 * R * (6 - fw) / 3 if fw < 6 else 0.0
        if hot and pulse > 0.3:
            g_amt = max(g_amt, 22 * R * pulse)
        if nxt is not None and hot and nxt - F <= 4:       # 碎裂退场：切片向两侧散开、渐隐
            ks = 1 - (nxt - F) / 4.0
            g_amt = max(g_amt, 90 * R * ks)
            im_w = im_w.copy(); im_w.putalpha(im_w.getchannel("A").point(lambda v: int(v * (1 - 0.7 * ks))))
        if g_amt > 0:
            im_w = slices(im_w, g_amt, seed + F // 2); im_sh = slices(im_sh, g_amt * 0.7, seed + 1 + F // 2)
        if pulse > 0.05:
            sp = int(16 * R * pulse)
            for col, dx in ((acc_now["main"], -sp), (acc_now["sec"], sp)):
                put(sub, word_img(ctx, word, size, fill=tuple(col) + (200,))[0], x - pad + dx, y - pad, scale=sc)
        sx_ = 1.0 + 0.07 * cl((fw - 9) / max(1.0, dur_w * FPS - 9)) if slow else 1.0      # 拖长音：字距感的横向慢拉
        put(sub, im_sh, x - pad + 8 * R, y - pad + 8 * R, scale=sc, sx=sx_)
        put(sub, im_w, x - pad, y - pad, scale=sc, sx=sx_)
    dd = ImageDraw.Draw(sub)                               # 计数小标 + 细线（Raf0x00 式）
    fm = font(L.F_MONO, 11 * R)
    lx, ly = x, y + th_ + 6 * R
    dd.rectangle([lx, ly, lx + min(tw, 260 * R) * T.expo_out(fw / 6.0), ly + 3 * R], fill=acc_now["main"] + (255,))


def lyric_layer(ctx, F, acc_now):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for i, ent in ctx.c["lyrics"].items():
        style, pos = ent["style"], ent.get("pos")
        r = LYR[i]
        Fs, Fe = fr(r["start"] / 1000.0), fr(lyr_end(i))
        if not (Fs <= F < Fe):
            continue
        f = F - Fs
        out_a = cl((Fe - F) / 5.0)                         # 末 5 帧淡出
        text = r["text"]
        sub = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if style == "hud":
            pos = pos or (0.06, 0.80)
            cs_ = ctx.cur_shot
            hb = ctx.cur_box
            if cs_ is not None and cs_.get("hud") and hb is not None:      # 让开 HUD v5 的信息块
                tgt_low = (hb[1] + 0.4 * (hb[3] - hb[1])) >= H * 0.55
                tgt_left = (hb[0] + hb[2]) / 2 < W / 2
                hk = cs_["hud"]
                if hk == "np":
                    pos = (0.06, 0.84) if tgt_low else (0.06, 0.12)
                elif hk == "type":
                    pos = (0.52, 0.86)
                elif hk in ("dossier", "lock", "scope", "codex", "blueprint"):
                    pos = (0.06, 0.84) if tgt_left else (0.52, 0.84)
                elif hk == "circle":
                    pos = (0.06, 0.84)
            L.lyric_hud(sub, text, f, 0, (W * pos[0], H * pos[1]), size=48, seed=i, acc=acc_now)
        elif style in ("serif_dark", "serif_ink"):
            col = (242, 238, 230, 255) if style == "serif_dark" else L.INK + (255,)
            x0, y0 = (W * 0.05, H * 0.70) if style == "serif_dark" else (W * 0.05, H * 0.84)
            size = int((66 if style == "serif_dark" else 60) * R)
            gl, tw, gh = glyphs(ctx, text, size, col, L.F_JPS, L.F_WIDE, 0.62)
            sh = glyphs(ctx, text, size, (0, 0, 0, 150) if style == "serif_dark" else (236, 232, 224, 140), L.F_JPS, L.F_WIDE, 0.62)[0]
            for j, ((im, x, ch), (ims, _, _)) in enumerate(zip(gl, sh)):   # 逐字从下方升起（16R，12 帧，字间 2 帧）
                k = eo((f - 2 * j) / 12.0)
                put(sub, ims, x0 + x + 2 * R, y0 + (1 - k) * 16 * R + 2 * R, alpha=k)    # 投影：换底色时也读得清
                put(sub, im, x0 + x, y0 + (1 - k) * 16 * R, alpha=k)
            meta = "LYR %02d  //  %06.3f — %06.3f" % (i, r["start"] / 1000.0, r["end"] / 1000.0)
            dcol = (196, 192, 184, 255) if style == "serif_dark" else (44, 42, 40, 255)
            ImageDraw.Draw(sub).text((x0, y0 + gh + 4 * R), meta, font=font(L.F_MONO, 11 * R), fill=dcol)
            ImageDraw.Draw(sub).rectangle([x0, y0 - 10 * R, x0 + 70 * R * eo(f / 8.0), y0 - 7 * R], fill=acc_now["main"] + (255,))
        elif style == "burst":
            burst_layer(ctx, sub, i, F, acc_now, ent.get("hot", False), ent.get("wall", False))
        if out_a < 1:
            sub.putalpha(sub.getchannel("A").point(lambda v: int(v * out_a)))
        lay.alpha_composite(sub)
    return lay


def cur_box(ctx, F):
    s = next((s for s in ctx.c["shots"] if s["Fa"] <= F < s["Fb"]), ctx.c["shots"][-1])
    a = ctx.ana[s["id"]]
    i = int(np.clip(F - a["F_in"], 0, len(a["accs"]) - 1))
    if "boxes" in a:
        return a["boxes"][i]
    if a.get("faces0"):
        b = a["faces0"][0]; cx, cy = a["face_c"][i][0]
        return (cx - (b[2] - b[0]) / 2, cy - (b[3] - b[1]) / 2, cx + (b[2] - b[0]) / 2, cy + (b[3] - b[1]) / 2)
    return None


def render_frame(ctx, F):
    c = ctx.c
    ctx.cur_box = cur_box(ctx, F)
    ctx.cur_shot = next((s for s in c["shots"] if s["Fa"] <= F < s["Fb"]), None)
    tr = next((t for t in c["trans"] if t["F"] - t["pre"] <= F < t["F"] + t["post"]), None)
    if tr is None:
        s = next((s for s in c["shots"] if s["Fa"] <= F < s["Fb"]), c["shots"][-1])
        img, acc = styled(ctx, s, F)
    else:
        sa = next(s for s in c["shots"] if s["Fb"] == tr["F"]); sb = next(s for s in c["shots"] if s["Fa"] == tr["F"])
        A, acc_a = styled(ctx, sa, F); B, acc_b = styled(ctx, sb, F)
        if tr["kind"] in ("glitch", "glitch_slam", "gcut"):
            k = (F - tr["F"]) / float(tr["pre"] if F < tr["F"] else tr["post"])
            if tr["kind"] == "glitch_slam" and F >= tr["F"]:
                B = T.slam(B, (F - tr["F"]) / float(tr["post"] + 6))
            img = T.glitch(A, B, k, F, acc_b, seed=tr["F"], amp=0.7 if tr["kind"] == "gcut" else 1.0)
            acc = acc_a if F < tr["F"] else acc_b
        else:
            n = tr["pre"] + tr["post"]; f = F - (tr["F"] - tr["pre"])
            if tr["kind"] == "lock_zoom":                  # 锁 A 的眼睛一带：目标框中心往上 10%（C 镜头用脸心）
                a = ctx.ana[sa["id"]]
                ii = int(np.clip(F - a["F_in"], 0, len(a["accs"]) - 1))
                if "boxes" in a:
                    bx = a["boxes"][ii]; tgt = ((bx[0] + bx[2]) / 2, bx[1] + (bx[3] - bx[1]) * 0.4)
                elif a.get("face_c") is not None and a["face_c"].shape[1]:
                    tgt = tuple(a["face_c"][ii][0])
                else:
                    tgt = (W / 2, H * 0.42)
                img = T.lock_zoom(A, B, f, n, tgt, acc_b)
            elif tr["kind"] == "hex":                      # EVA 蜂巢转场（用户第 5 条）
                a = ctx.ana[sa["id"]]
                ii = int(np.clip(F - a["F_in"], 0, len(a["accs"]) - 1))
                if "boxes" in a:
                    bx = a["boxes"][ii]; tgt = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
                else:
                    tgt = (W / 2, H / 2)
                img = HX.hex_wipe(A, B, f, n, tgt, acc_b, seed=tr["F"])
            elif tr["kind"] == "slit":                     # 画幅收窄 + 强调字（用户："IMPERSONATOR 这个…我很喜欢"）
                busy = any(e_["style"] == "burst" and fr(LYR[li]["start"] / 1000.0) <= F < fr(lyr_end(li))
                           for li, e_ in c["lyrics"].items())
                img = V4.slit(A, B, f, n, tr["pre"], acc_b, text=None if busy else tr.get("text"), sub=None if busy else tr.get("sub"))
                if tr.get("tape"):                         # v10：缝两侧黑边上拉警戒线（用户："还可以加一些警戒线"）
                    img = TP.tapes(img, f, n, tr["pre"], dict(acc_b, main=tuple(tr.get("tape_col", acc_b["main"]))), tr["tape"], seed=tr["F"], mode="slit")
            elif tr["kind"] == "tape":                     # v10：警戒线转场（三条交叉胶带，切点在胶带后面）
                if f < tr["pre"]:
                    img = A
                else:
                    s_ = 1.04 - 0.04 * T.expo_out((f - tr["pre"]) / float(max(1, n - tr["pre"])))
                    img = cv2.warpAffine(B, np.float32([[s_, 0, W / 2 * (1 - s_)], [0, s_, H / 2 * (1 - s_)]]), (W, H), flags=cv2.INTER_LINEAR)
                img = TP.tapes(img, f, n, tr["pre"], dict(acc_b, main=tuple(tr.get("tape_col", acc_b["main"]))), tr.get("tape") or ["CAUTION"], seed=tr["F"], mode="cross")
            elif tr["kind"] == "whip":
                img = T.whip(A, B, f, n, direction=tr.get("dir", 1))
            elif tr["kind"] == "grid":
                img = T.grid(A, B, f, n, acc=acc_b)
            else:
                img = T.panel_slap(A, B, f, n, acc_b, titles=sb.get("titles", []) + ["FILE 03", "FILE 04", "FILE 05", "FILE 06"],
                                   merge_from=tr["pre"])
            acc = L.mix_acc(acc_a, acc_b, f / float(n))
    g = glitch_amp(c, F)
    if g > 0.01:                                           # 全局 glitch 包络（蓄力 / 换气 / 爆点）
        img = T.glitch(img, img, 0.0, F, acc, seed=31, amp=g)
    for w_ in c.get("cine", []):                           # v10：break 的 21:9 遮幅 + 电影质感（进出渐变）
        k_ = CN.ramp(F / FPS, w_)
        if k_ > 0:
            img = CN.finish(img, F, k=k_, bars=k_)
    sb_ = next((s for s in c["shots"] if s.get("bars") and s["Fa"] <= F < s["Fb"]), None)
    if sb_ is not None:                                    # 换气段上下黑边收拢（爆点那一帧弹开）
        bh = int(H * sb_["bars"] * T.expo_io((F - sb_["Fa"]) / float(sb_["Fb"] - sb_["Fa"]) * 1.4))
        if bh > 0:
            img = img.copy(); img[:bh] = (6, 5, 6); img[H - bh:] = (6, 5, 6)
    out = L.compose(img, lyric_layer(ctx, F, acc))
    if c.get("captions"):                                  # v9 成片：前奏咏唱 / 间奏招式名
        cap = TT.caption_layer(c["captions"], F, acc)
        if cap is not None:
            out = L.compose(out, cap)
    for t0, t1 in c.get("fade", []):                      # 片尾淡出到黑（每帧 ≤ 约 4 亮度）
        if F / FPS >= t0:
            k = cl((F / FPS - t0) / (t1 - t0))
            out = (out.astype(np.float32) * (1 - k)).astype(np.uint8)
    return out


_CTX = None
NV_ENC = ["-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "constqp", "-qp", "12", "-pix_fmt", "yuv444p", "-profile:v", "rext"]


def nframes(path):
    r = subprocess.run([os.path.join(os.path.dirname(SF.FF), "ffprobe.exe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return -1


def _work(args):
    """一个分块：逐帧渲染，原始帧直接喂给显卡编码（NVENC HEVC 4:4:4 近无损），不落 PNG。"""
    global _CTX
    key, ci, chunk = args
    cv2.setNumThreads(2)                                   # 用户：CPU 占用别太高
    if _CTX is None:
        _CTX = Ctx(prep(key))
    c = _CTX.c
    out = os.path.join(c["work"], "chunks", "c%03d.mkv" % ci)
    if os.path.exists(out) and nframes(out) == len(chunk):
        return 0
    cmd = [SF.FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-"] + NV_ENC + [out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    damp = {F for s_ in c["shots"] if s_.get("damp") for F in range(s_["Fa"] - 2, s_["Fb"] + 3)}
    prev = None
    for F in chunk:
        img = render_frame(_CTX, F)
        if F in damp and prev is not None:                 # 源里黑白骤变的镜头：亮度跳变 > 30 时和上一帧混合（3–4 帧光拖尾），压到 QC 闪烁阈值以下
            d = abs(float(img.mean()) - float(prev.mean()))
            if d > 30:
                k = min(0.6, 0.3 + (d - 30) / 60.0)
                img = (img.astype(np.float32) * (1 - k) + prev.astype(np.float32) * k).astype(np.uint8)
        prev = img
        p.stdin.write(np.ascontiguousarray(img).tobytes())
    p.stdin.close(); p.wait()
    return len(chunk)


def render(c, jobs=int(os.environ.get("RENDER_JOBS", "5")), per=360):
    os.makedirs(os.path.join(c["work"], "chunks"), exist_ok=True)
    frames = list(range(c["F0"], c["F1"]))
    chunks = [(c["key"], k, frames[i:i + per]) for k, i in enumerate(range(0, len(frames), per))]
    t = time.time(); done = 0
    with Pool(min(jobs, len(chunks))) as p:
        for n in p.imap_unordered(_work, chunks, chunksize=1):
            done += n
            print("rendered %d / %d frames  %.0fs" % (done, len(frames), time.time() - t), flush=True)
    bad = [k for _, k, ch in chunks if nframes(os.path.join(c["work"], "chunks", "c%03d.mkv" % k)) != len(ch)]
    print("render %.0fs, 不完整分块: %s" % (time.time() - t, bad or "无"))


def encode(c):
    """分块拼接（帧数逐块核过）+ 原曲 → 显卡 H.264 成片。"""
    import glob
    wav = [q for q in glob.glob(r"D:\备份\*.wav") if "PROVANT" in os.path.basename(q)][0]
    out = os.path.join(ROOT, "out", {"V5": "v3", "V6": "v4", "V7": "v5", "V8": "v6", "V9": "v7", "V10": "v8", "V11": "v9", "V12": "v10"}.get(c["key"], "v2"), c["out"]); os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lst = os.path.join(c["work"], "chunks", "list.txt")
    ks = sorted(glob.glob(os.path.join(c["work"], "chunks", "c*.mkv")))
    open(lst, "w").write("".join("file '%s'\n" % k.replace(os.sep, "/") for k in ks))
    venc = ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "16", "-b:v", "0", "-profile:v", "high", "-pix_fmt", "yuv420p"]
    if c.get("song_map"):                                  # v9 成片：成片比歌长，只出画面（混音由 v4_audio 合成；带原曲 + -shortest 会把片尾截掉）
        cmd = [SF.FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-map", "0:v"] + venc + [out]
    else:
        cmd = [SF.FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
               "-ss", "%.4f" % (c["F0"] / FPS), "-t", "%.4f" % ((c["F1"] - c["F0"]) / FPS), "-i", wav,
               "-map", "0:v", "-map", "1:a"] + venc + ["-c:a", "aac", "-b:a", "320k", "-shortest", out]
    subprocess.run(cmd, check=True)
    print("->", out, nframes(out), "frames")


def sheet(c, n=20):
    picks = np.linspace(c["F0"] + 2, c["F1"] - 2, n).astype(int)
    if c["key"] == "B":                                    # 高潮：换气 / 爆点附近多取几帧
        picks = sorted(set(list(np.linspace(c["F0"] + 2, fr(127.5), 8).astype(int)) +
                           [fr(127.6), fr(127.9), fr(128.128), fr(128.128) + 1, fr(128.128) + 2, fr(128.128) + 5, fr(128.128) + 12,
                            fr(128.9), fr(129.5), fr(130.1), fr(130.8), c["F1"] - 3]))
    ims = []
    mp4 = os.path.join(ROOT, "out", {"V5": "v3", "V6": "v4", "V7": "v5", "V8": "v6", "V9": "v7", "V10": "v8", "V11": "v9", "V12": "v10"}.get(c["key"], "v2"), c["out"])
    for F in picks:
        pr = subprocess.run([SF.FF, "-v", "error", "-ss", "%.4f" % ((F - c["F0"]) / FPS), "-i", mp4, "-frames:v", "1", "-vf", "scale=640:360",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
        im = np.frombuffer(pr.stdout, np.uint8)[:640 * 360 * 3].reshape(360, 640, 3).copy()
        cv2.putText(im, "%.3f" % (F / FPS), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
        ims.append(im)
    while len(ims) % 4:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(os.path.join(ROOT, "ae", "qa", "v2style", "_preview_%s_sheet.jpg" % c["key"]),
                np.vstack([np.hstack(ims[i:i + 4]) for i in range(0, len(ims), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    key = sys.argv[1]
    what = sys.argv[2] if len(sys.argv) > 2 else "all"
    c = prep(key)
    if what in ("analyze", "all"):
        t = time.time()
        todo = [(key, s) for s in c["shots"] if "--redo" in sys.argv or not ana_ok(c, s)]
        todo.sort(key=lambda a: -(a[1]["look"] in ("Cd", "Cb")))   # 抠像的长活先派
        with Pool(min(int(os.environ.get("ANA_JOBS", "4")), max(1, len(todo)))) as p:
            for k, r in enumerate(p.imap_unordered(analyze_shot, todo, chunksize=1)):
                print("analyzed %d/%d" % (k + 1, len(todo)), r, "%.0fs" % (time.time() - t), flush=True)
        print("analyze %.0fs" % (time.time() - t))
    if what in ("render", "all"):
        render(c)
    if what in ("encode", "all"):
        encode(c)
    if what in ("sheet", "all"):
        sheet(c)

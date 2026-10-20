# -*- coding: utf-8 -*-
"""
PROVANT v8 成片 EDL（v10）→ data/edl_v10.json（.venv_gpu python scripts/v2_preview.py V10 all）
用户（10-20）："原来的剪辑切的太碎，这一版把对应的角色的片段放在一块，原先的版本有些角色放宝具被掐断，然后打斗场面也被掐断。
另外 fate Apocrypha 素材少用，多换点 FGO 的素材进去"
→ 按角色分块：每块是同一来源里连续的一段，块内切点 = 源片自己的剪辑点（硬切），整块拟合一个速度（约 0.85–1.15），
  宝具 / 打斗整段放完；块与块之间才用 slit / glitch 转场；块内切点离拍点 <= 0.09 s 的吸到拍上。
  HUD 只上块开场、宝具、关键镜头（其余 R 镜头 nohud，只留 r_base 底色 + 全局 glitch + 歌词）。
  Apocrypha 只剩用户认可过的预览 B（124.2–132.2）里 4 个镜头；其余换成 FGO（伊什塔尔 / 帝都聖杯奇譚 CM / 贞德 / 卡美洛 /
  武藏 CM / 阿拉什 / 魁札尔 / 阿尔托莉雅 / 马修）+ HF（士郎 vs 言峰、Rider 天马）。
风格层全部复用 v9（R / Cd / Cb / P / M / A / G / B / Cs、人物配色、glitch 包络、强拍脉冲、歌词、台词混音）。
片头（0–8.85）、换气（77.0–79.3）、预览 B、终幕 L0076、片尾马修沿用 v9 原镜头（预览 B 的 KK002 换 KK018：KK002 进了士郎块）。
输出：data/edl_v10.json、EDL_v10.md、ae/qa/v10/<块>.jpg
"""
import json, os, re, subprocess, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import edl_v9 as V9
import palette as PALT
from edl_v9 import ROOT, FF, FPS, SONG_END, B, POOL, H4

_bt0 = V9.bt


def bt(b):
    return float(b) if isinstance(b, float) else _bt0(b)


V9.bt = bt                                                  # make_shot 里按拍号取时间：块内镜头直接给秒数
INFO = H4.INFO
V9.SLIT_NAME.update(ishtar="ISHTAR", arash="ARASH", bedivere="BEDIVERE", lion_king="LION KING", quetzalcoatl="QUETZALCOATL")
OLD = {s["id"]: s for s in json.load(open(os.path.join(ROOT, "data", "edl_v9.json"), encoding="utf-8"))["shots"]}
OLD_TR = json.load(open(os.path.join(ROOT, "data", "edl_v9.json"), encoding="utf-8"))["trans"]


def slit(key=None, text=None, sub=None):
    if key:
        inf = INFO[key]
        return dict(kind="slit", text=V9.SLIT_NAME[key], sub="CLASS  %s   /   %s" % (inf["cls"], inf.get("jp", "")))
    return dict(kind="slit", text=text, sub=sub)


# ---------------------------------------------------------------- 块
# items：代码 或 (代码, 选项)。选项：look / hud（模板名或 "auto"；R 不写 = nohud）/ nohud（P 用）/ who（人物改写，HUD 和配色用）/
#        s0 s1（源区间，秒）/ take（从头取 N 秒）/ tail（取最后 N 秒）/ w（慢放权重，<1 = 比块速慢）/ at（此镜头从这个拍号 / 秒开始）
#        其余（impact、slam、burst、hold、titles、manga、tone、caption、tag、sfx …）原样给 make_shot
FIX = lambda *ids: dict(fixed=list(ids))
BLOCKS = [
    dict(id="V01", name="INTRO", **FIX("E01_00", "E01_01", "E01_02", "E01_03", "E01_04", "E01_05"), kind="intro"),
    # 8.85–28.10 伊什塔尔（FGO 巴比伦尼亚）：飞行 → 金色光束齐射 → 草原连环爆炸 → 击落狮形魔兽 → 城墙上飞踢 → 落地
    dict(id="V02", name="ISHTAR", t0=15, t1=48, kind="battle", enter=slit("ishtar"),
         items=[("O1002", dict(s1=3.40, hud="lock", who="伊什塔尔")), ("O1003", dict(s1=3.80)),      # 源里光束频闪：避开（timeline_flash）
                ("O1004", dict(s1=4.48, hud="np", who="伊什塔尔")), "O1005",
                ("O1006", dict(s1=8.45, look="A", who="伊什塔尔", sfx="BOOM!")), "O1007", ("O1008", dict(look="M", manga="band")),
                "O1009", "O1010", "O1011", "O1012", "O1013", ("O1014", dict(hud="scope")),
                ("O1018", dict(look="A", sfx="KRAK!")),
                ("O1019", dict(take=3.6, hud="dossier"))]),
    # 28.10–38.36 帝都聖杯奇譚（慢唱 JUST LIKE ALL THE REST）：用户标注的 OA 7–27 s 全用（v10b 补 OA001 / OA002）
    #   夜空巨蛇 → 狂笑 → 血月 → 月夜帝都 → 相握的手 → 樱月 → 冲田〔Alter〕灵基卡（OA013 前半是衣摆，从 25.35 起）
    dict(id="V03", name="OKITA ALTER", t0=48, t1=65, kind="lyrical", enter=slit("okita_alter"),
         items=[("OA001", dict(s0=8.5, w=0.65)), ("OA002", dict(w=0.65)), ("OA003", dict(w=0.65)), ("OA009", dict(w=0.65)),
                ("OA010", dict(w=0.65, look="M", manga="band", caption="帝都聖杯奇譚", tag="TEITO // HOLY GRAIL")),
                ("OA011", dict(w=0.65)), ("OA012", dict(w=0.65)), ("OA013", dict(s0=25.35, w=0.5))]),     # 灵基卡窗口里只有衣摆（脸到最后几帧才转过来）→ 原画
    # 38.36–51.25 冲田 vs 冲田〔Alter〕CM 打斗原样（帝都 → 京都和室 → 雨夜）；OA014 延到用户标注的 27.0；OY 19–23 全用（补 OY003）
    dict(id="V04", name="OKITA", t0=65, t1=87, kind="battle", enter=dict(kind="tape", key="okita"),     # v10：警戒线转场
         items=[("OA004", dict(hud="lock")), "OA005", "OA006", ("OA007", dict(look="M", manga="impact")), ("OA008", dict(look="Cd")),
                ("OA014", dict(s1=26.98, hi=27.0, hud="np")), ("OJ001", dict(hud="type")), "OJ002", ("OJ003", dict(look="Cd")), "OY001",
                "OY003", ("OY002", dict(w=0.6, look="G"))]),
    # 51.25–59.56 贞德（FGO）：骑士群中 → 持旗立像 → "Fist emblazoned"（57.24）挥拳冲锋
    dict(id="V05", name="JEANNE", t0=87, t1=101, kind="battle", enter=dict(kind="tape", key="jeanne"),     # 片头已有 JEANNE 缝 → 警戒线
         items=[("L0010", dict(hud="dossier")), ("L0040", dict(look="M", manga="story")),
                ("CF042", dict(at=97, hud="type", impact=3, slam=10, burst=True)), "CF043"]),     # CF044 在片头灵基卡里（src 85.77 起）
    # 59.56–77.02 魁札尔·科亚特尔 Ultimo Tope Patada（FGO 巴比伦尼亚，用户 v10："羽蛇神飞踢多给点镜头"）：
    #   火球从天而降 → 速度线 → 摔角靴 → 提亚马特 → 火环 → 翻身 → 展臂 → 74.70 "Judge his master" 飞踢命中 → 陨坑爆燃
    dict(id="V06", name="QUETZALCOATL", t0=101, t1=131, kind="battle", hot=True, enter=slit("quetzalcoatl"),
         items=[("DE002", dict(hud="lock", who="魁札尔·科亚特尔")), "DE003", "DE004", "DE005", "DE006", "DE008",
                ("DE009", dict(look="A", sfx="WHAM!", who="魁札尔·科亚特尔")), ("DE010", dict(look="Cd")), "DE011", ("DE012", dict(hud="hexwall")),
                "DE013", ("DE014", dict(look="A", sfx="BOOM!")), "DE015", ("DE017", dict(s1=16.05)), ("DE018", dict(who="魁札尔·科亚特尔", hud="type")),
                ("DE019", dict(s1=18.25)), ("DE032", dict(who="魁札尔·科亚特尔")),
                ("DE033", dict(at=127, look="P", hold=0, impact=3, slam=10, burst=True, hud="np", who="魁札尔·科亚特尔")),
                ("DE034", dict(s1=39.09)), ("DE038", dict(take=1.7, look="P", hold=0, nohud=True))]),
    dict(id="V07", name="BREATH", **FIX("E06_00"), kind="breath"),
    # 79.34–98.01 卡美洛（FGO 剧场版）：狮子王 → 圣枪 Rhongomyniad 整段 → 贝德维尔 → 96.85 "And cast the first die!" 金剑举起
    dict(id="V08", name="CAMELOT", t0=135, t1=167, kind="battle",
         enter=slit(text="CAMELOT", sub="第六特異点   /   神聖円卓領域キャメロット"),
         items=["2J025", ("2J026", dict(look="A", sfx="SLASH!")), ("2J027", dict(hud="type", who="狮子王")), "2J028", "2J029", "2J030",
                ("2J031", dict(hud="np", who="狮子王")), ("2J032", dict(look="P", hold=0, nohud=True)), "2J033",
                ("2J034", dict(hud="lock", who="贝德维尔")), "2J035",
                ("2J036", dict(at=165, s1=42.0, look="P", hold=0, impact=3, slam=10, burst=True, hud="lock", who="贝德维尔"))]),
    # 98.01–109.69 武藏 CM 整支（用户 v10："宫本武藏也多一点"；素材只有这一支 15 s CM，竹林开头 + 眼部特写也补上）；
    #   参りましょう 按源片位置同步（落在 100.89 之后的不唱段）
    dict(id="V09", name="MUSASHI", t0=167, t1=187, kind="lyrical", enter=slit("musashi"),
         items=[("MU001", dict(s0=0.4, lo=0.35, hud="type")), ("MU002", dict(look="M", manga="band")), ("MU003", dict(hud="np")), "MU004",
                ("MU005", dict(s1=11.96, hi=11.97, w=0.8, look="G"))]),
    # 109.69–124.20 阿拉什 流星一条（Stella）整段：持弓 → 拉满 → 怒吼 → 放箭 → 光箭升空 → 螺旋光爆
    dict(id="V11", name="ARASH", t0=187, t1=212, kind="build", hot=True, enter=slit("arash"),
         items=[("QI010", dict(hud="lock", who="阿拉什")), ("QI012", dict(hud="np", who="阿拉什")),
                ("QI015", dict(look="A", who="阿拉什", sfx="FWOOSH!")), "QI016", ("QI017", dict(take=3.5)),
                ("QI018", dict(look="P", hold=0, nohud=True))]),
    dict(id="V12", name="PREVIEW B", fixed=["E10_09", "E10_10", "KK018", "E10_12", "E10_13", "E10_14", "E10_15",
                                            "E11_00", "E11_01", "E11_02", "E11_03", "E11_04"], kind="build", enter=dict(kind="glitch")),
    # 132.21–152.02 Saber Alter vs Rider（HF3，用户 v10："saber alter 也多一点"）：Saber Alter 冷笑 → 魔力翼斩 → 突进 → 交锋 → 缠斗 →
    #   142.11 Rider 魔力包围 → 伸掌 → 145.61 "Fist emblazoned" 骑英之缰绳 → 天马 → 爆炸 → 红雷
    dict(id="V13", name="SABER ALTER", t0=226, t1=260, kind="climax", hot=True, enter=slit("artoria_alter"),
         items=[("RV013", dict(look="G")), ("RV014", dict(hud="lock")), ("RV015", dict(s1=19.33)), ("RV017", dict(look="M", manga="impact")),
                ("RV019", dict(look="A", sfx="CLANG!")), ("RV021", dict(take=1.0)),
                ("RV094", dict(at=243, s0=125.32, s1=126.65, look="P", hold=0, impact=3, slam=10, burst=True, hud="lock")),
                ("RV097", dict(tail=2.2, hud="np")),
                ("RV098", dict(at=249, look="P", hold=0, impact=3, slam=10, burst=True, hud="type")), "RV099", "RV100",
                ("RV101", dict(look="A", sfx="FWOOSH!")), ("RV102", dict(s0=137.67, s1=138.42, look="P", hold=0, nohud=True)), ("RV103", dict(take=1.95))]),     # RV102 源内 137.6 黑帧 + 138.5 闪
    # 152.02–163.68 阿尔托莉雅（FGO Memorial）：背影 → 星之宝石 → 仰望（灵基卡）→ 飞越云海 → 圆桌骑士 → 面部 → 持剑飞掠
    dict(id="V15", name="ARTORIA", t0=260, t1=280, kind="climax", hot=True, enter=slit("artoria"),
         items=[("PM066", dict(take=2.0, hud="type")), "PM068", ("PM071", dict(w=0.7, look="G")), "PM072", "PM073",
                ("PM075", dict(look="A", sfx="FWOOSH!")), ("PM076", dict(look="M", manga="band")), ("PM077", dict(hud="np")), "PM078"]),
    dict(id="V16", name="FINALE", **FIX("E11_39"), kind="climax", hot=True, enter=dict(kind="cut")),
    dict(id="V17", name="MASH", **FIX("E12_00"), kind="outro"),
]
GLITCH = V9.GLITCH + [(56.6, 57.237, 0.1, 0.6), (57.237, 57.38, 1.0, 0.0)]      # 贞德 "Fist emblazoned"
PUNCH = sorted(V9.PUNCH + [57.237])
DIALOG_FIX = [d for d in V9.DIALOG if d["yid"] != "MUkengo4K60"]


def item(spec):
    code, o = (spec, {}) if isinstance(spec, str) else (spec[0], dict(spec[1]))
    r = POOL[code]
    if code.startswith("L0"):
        r = V9.upgrade(r)                                   # L0（1080）→ PM（1440）同画面
    lo, hi = o.pop("lo", r["in"] + 1 / 60.0), o.pop("hi", r["out"] - 1 / 60.0)     # 用户标注区间比池子镜头长时可放宽
    s0, s1 = o.pop("s0", lo), o.pop("s1", hi)
    if "take" in o:
        s1 = s0 + o.pop("take")
    if "tail" in o:
        s0 = s1 - o.pop("tail")
    return dict(code=code, r=r, s0=s0, s1=s1, lo=lo, hi=hi, o=o)


def grid():
    g = list(B) + [(B[k] + B[k + 1]) / 2 for k in range(len(B) - 1)]
    return np.array(sorted(g))


GRID = grid()


def place(blk):
    """块内排时：按锚点分段，每段源时长 / 权重之和摊到段长上（= 整段一个速度）；块内切点离拍点 <= 0.09 s 的吸到拍上。"""
    t0, t1 = bt(blk["t0"]), bt(blk["t1"])
    its = [item(x) for x in blk["items"]]
    marks = [(0, t0)] + [(k, bt(it["o"].pop("at"))) for k, it in enumerate(its) if "at" in it["o"]] + [(len(its), t1)]
    anchored = {k for k, _ in marks}
    for (k0, ta), (k1, tb) in zip(marks[:-1], marks[1:]):
        seg = its[k0:k1]
        if not seg:
            continue
        D = sum((it["s1"] - it["s0"]) / it["o"].get("w", 1.0) for it in seg)
        k = D / (tb - ta)
        t = ta
        for it in seg:
            d = (it["s1"] - it["s0"]) / it["o"].get("w", 1.0) / k
            it["a"], it["b"] = t, t + d
            t += d
        seg[-1]["b"] = tb
    for j in range(1, len(its)):
        if j in anchored:
            continue
        c = its[j]["a"]
        g = float(GRID[np.argmin(np.abs(GRID - c))])
        if abs(g - c) <= 0.09 and g - its[j - 1]["a"] >= 0.2 and its[j]["b"] - g >= 0.2:
            its[j - 1]["b"] = its[j]["a"] = g
    for it in its:
        it["speed"] = (it["s1"] - it["s0"]) / (it["b"] - it["a"])
        lim = 0.5 if it["r"]["res"] >= 1440 else 0.6
        if it["o"].get("hold") and it["o"].get("look") == "P" and it["s1"] - it["s0"] < 0.5:
            lim = 0.0                                       # 定格版画：本来就是步进取帧
        if not (lim <= it["speed"] <= 1.5):
            print("  ! %s %s 速度 %.2f" % (blk["id"], it["code"], it["speed"]))
    return its


def shot_of(it, blk, idx):
    o = dict(it["o"])
    r = dict(it["r"])
    if "who" in o:
        r["who"] = o.pop("who")
    look = o.pop("look", "R")
    nohud = o.pop("nohud", False)
    o.pop("w", None)
    hud = o.pop("hud", None)
    pin = dict(o, src_a=round(it["s0"], 4), speed=round(it["speed"], 4), lo=round(it["lo"], 4), hi=round(it["hi"], 4))
    if hud and hud != "auto":
        pin["hud"] = hud
    seg = dict(seg=blk["id"], kind=blk["kind"])
    sh = V9.make_shot(r, round(it["a"], 4), round(it["b"], 4), pin, seg, look)
    sh["a"], sh["b"] = round(it["a"], 4), round(it["b"], 4)
    sh["id"] = "%s_%02d" % (blk["id"], idx)
    sh["seg"] = blk["id"]
    sh["hot"] = bool(blk.get("hot"))
    sh["block"] = blk["name"]
    if look in ("R", "P") and (nohud or (look == "R" and not hud)):
        sh["nohud"] = True
    return sh


# 用户 10-20："像 0:00:25 这种将伊什塔尔场景遮罩的处理不要用（不要把场景完全遮起来）" →
# 不用 Cb（整个环境换成档案纸）；Cd 只留给脸占满画面的特写；预览 B 的 A1033 档案页改 R，进它的 panel_slap 改 glitch
FIXED_OVR = {"E11_04": dict(look="R", hud="type")}


def fixed_shot(ref, blk):
    """沿用 v9 的镜头（同 id → 分析缓存可复用）；预览 B 的 KK002 换成 KK018（同类拳击极近特写）。"""
    if ref in FIXED_OVR:
        old = OLD[ref]; ov = FIXED_OVR[ref]
        pin = dict(src_a=old["src_a"], speed=old["speed"], lo=old["lo"], hi=old["hi"], hud=ov["hud"])
        sh = V9.make_shot(POOL[old["code"]], old["a"], old["b"], pin, dict(seg=old["seg"], kind="build"), ov["look"])
        sh.update(a=old["a"], b=old["b"], id=ref, seg=old["seg"], hot=old.get("hot", True))
    elif ref in OLD:
        sh = dict(OLD[ref])
    else:
        old = OLD["E10_11"]
        pin = dict(V9.PB[214], code=ref)
        sh = V9.make_shot(POOL[ref], old["a"], old["b"], pin, dict(seg="V12", kind="build"), "R")
        sh["a"], sh["b"] = old["a"], old["b"]
        sh["id"] = "E10_11"
        sh["seg"] = "E10"; sh["hot"] = True
    sh["block"] = blk["name"]
    return sh


def build():
    V9._NM["n"] = 0
    shots, segs_of, bounds = [], [], []
    mus = None
    for blk in BLOCKS:
        start = len(shots)
        if "fixed" in blk:
            for ref in blk["fixed"]:
                shots.append(fixed_shot(ref, blk)); segs_of.append(blk["kind"])
        else:
            its = place(blk)
            for k, it in enumerate(its):
                sh = shot_of(it, blk, k)
                shots.append(sh); segs_of.append(blk["kind"])
                if it["code"] == "MU003":
                    mus = sh
        bounds.append((start, blk))
    for k in range(1, len(shots)):                          # 连续性自检
        if abs(shots[k]["a"] - shots[k - 1]["b"]) > 1e-3:
            print("  ! 时间不连续 %s %.3f → %s %.3f" % (shots[k - 1]["id"], shots[k - 1]["b"], shots[k]["id"], shots[k]["a"]))
    V9.assign_hud([s for s in shots if s["look"] in ("R", "P") and not s.get("nohud")])
    # ---- 转场：固定段内部沿用 v9；块内全部硬切；块边界按块的 enter（没写的按 v9 规则自动）
    auto = {round(t["t"], 3): t for t in V9.assign_trans(shots, segs_of)}
    trans = []
    fixed_ranges = [(shots[s]["a"], shots[s + len(b["fixed"]) - 1]["b"]) for s, b in bounds if "fixed" in b]
    for t in OLD_TR:
        if any(a + 1e-3 < t["t"] < b - 1e-3 for a, b in fixed_ranges):
            trans.append(dict(t, kind="glitch", pre=5, post=6) if t["kind"] == "panel_slap" else dict(t))     # 档案拍贴会盖住整个画面
    for s, blk in bounds:
        if s == 0:
            continue
        t = shots[s]["a"]
        en = blk.get("enter")
        if en is None:
            old = next((x for x in OLD_TR if abs(x["t"] - t) < 1e-3), None)
            if old:
                trans.append(dict(old))
            elif round(t, 3) in auto:
                trans.append(auto[round(t, 3)])
            continue
        if en["kind"] == "cut":
            continue
        pre, post = {"slit": (12, 18), "glitch_slam": (4, 8), "glitch": (5, 6), "whip": (6, 6), "hex": (9, 9), "tape": (10, 24)}[en["kind"]]
        tr = dict(en, t=t, pre=pre, post=post)
        if en["kind"] == "tape":
            key = tr.pop("key"); inf = INFO[key]
            tr["tape"] = ["CAUTION", inf["en"], "CLASS " + inf["cls"], inf.get("jp", "")]
            tr["tape_col"] = list(PALT.SIG[key][0])
        trans.append(tr)
    trans.sort(key=lambda x: x["t"])
    for tr in trans:                                        # v10：每个角色缝转场两侧都拉警戒线（用户："还可以加一些警戒线等画面处理方式"）
        if tr["kind"] == "slit":
            key = {v: k for k, v in V9.SLIT_NAME.items()}.get(tr["text"])      # 胶带颜色 = 名字对应角色的签名色（不跟入场镜头走）
            tr["tape_col"] = list(PALT.SIG[key][0]) if key else list(PALT.SIG["lion_king"][0])     # CAMELOT：狮子王金
            tr["tape"] = ["CAUTION", tr["text"]] + [re.sub(r"\s+", " ", x).strip() for x in tr.get("sub", "").split("/") if x.strip()]
    print("  转场:", dict(Counter(t["kind"] for t in trans)), " slit:", [t["text"] for t in trans if t["kind"] == "slit"])
    # ---- 台词：武藏 参りましょう 按 MU003 画面同步（源 6.976 s 出现的成片时刻）
    dialog = list(DIALOG_FIX)
    t_line = mus["a"] + (6.976 - mus["src_a"]) / mus["speed"]
    dialog.append(dict(t=round(t_line, 3), yid="MUkengo4K60", s0=6.96, s1=7.80, who="宫本武藏", note="MU003 同源画面同步"))
    print("  武藏台词 @ %.2f s" % t_line)
    assert 100.95 < t_line and t_line + 0.9 < 108.2, "武藏台词落进唱段"
    dialog.sort(key=lambda d: d["t"])
    fw = V9.flash_check(json.loads(json.dumps(shots)))      # 只报告不滑动：块内是连续源，滑了就断（成片按 timeline_flash 预检 + QC 处理）
    fw = [w for w in fw if not w[1].startswith("slide")]
    pinned = {s["code"].split(">")[0] for s in shots}
    dd = V9.dupes(shots, pinned)
    for s in shots:
        s.pop("_risky", None)
    lyr = V9.lyric_map(shots)
    cfg = dict(T0=0.0, T1=SONG_END, out="PROVANT_AMV_v8_video.mp4", shots=shots, trans=trans, lyrics=lyr,
               glitch=GLITCH + V9.beat_glitch(), punch=PUNCH, fade=V9.FADE, dialog=dialog, hud_v=5)
    json.dump(cfg, open(os.path.join(ROOT, "data", "edl_v10.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # ---- 统计 + 人看的表
    n = len(shots)
    lk = Counter(s["look"] + ("-" if s.get("nohud") else "") for s in shots)
    gp = Counter(s["group"] for s in shots)
    gpt = Counter()
    for s in shots:
        gpt[s["group"]] += s["b"] - s["a"]
    durs = [s["b"] - s["a"] for s in shots]
    Ls = ["# EDL v10（自动生成：scripts/edl_v10.py）", "",
          "镜头 %d（平均 %.2f s，中位 %.2f s）；风格 %s；转场 %s；系列镜头数 %s；系列时长 %s" % (
              n, float(np.mean(durs)), float(np.median(durs)), dict(lk), dict(Counter(t["kind"] for t in trans)), dict(gp),
              {k: round(v, 1) for k, v in gpt.items()}), ""]
    cur = None
    for s in shots:
        if s["block"] != cur:
            cur = s["block"]; Ls += ["", "## %s" % cur]
        Ls.append("- `%s` %7.3f–%7.3f %-3s%s %-10s x%.2f %s/%s %s — %s（%s）" % (
            s["id"], s["a"], s["b"], s["look"], "-" if s.get("nohud") else (s.get("hud", "") and ":" + s["hud"]), s["code"], s.get("speed", 1),
            s["group"], s["res"], s["kind"], s["what"][:40], s["who"][:16]))
    Ls += ["", "## 预检提示"] + ["- " + str(w) for w in fw] + ["- 画面重复：" + str(dd)]
    open(os.path.join(ROOT, "EDL_v10.md"), "w", encoding="utf-8").write("\n".join(Ls))
    print(Ls[2])
    for w in fw:
        print("  !", w)
    if dd:
        print("  ! 画面重复:", dd)
    return cfg


def sheets(cfg):
    from cards import src_path
    from PIL import Image, ImageDraw, ImageFont
    out = os.path.join(ROOT, "ae", "qa", "v10"); os.makedirs(out, exist_ok=True)
    fnt = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 15)
    by = {}
    for s in cfg["shots"]:
        by.setdefault(s["seg"], []).append(s)
    tr_at = {round(t["t"], 3): t["kind"] for t in cfg["trans"]}
    for seg, ss in by.items():
        cols = 6; rows = (len(ss) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 320, rows * 210), (16, 16, 18)); d = ImageDraw.Draw(sheet)
        for k, s in enumerate(ss):
            t = s["src_a"] + (s["b"] - s["a"]) * s.get("speed", 1) * 0.5
            p = subprocess.run([FF, "-v", "error", "-ss", "%.3f" % min(t, s["hi"]), "-i", src_path(s["yid"]), "-frames:v", "1",
                                "-vf", "scale=320:180", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
            if len(p.stdout) >= 320 * 180 * 3:
                sheet.paste(Image.frombytes("RGB", (320, 180), p.stdout[:320 * 180 * 3]), ((k % cols) * 320, (k // cols) * 210))
            x, y = (k % cols) * 320, (k // cols) * 210 + 181
            d.text((x + 4, y), "%s %s%s %s %.2f %s" % (s["id"][-2:], s["look"], "-" if s.get("nohud") else "", s["code"], s["a"],
                                                      tr_at.get(round(s["a"], 3), "|")), font=fnt, fill=(255, 220, 90))
        sheet.save(os.path.join(out, "%s.jpg" % seg), quality=85)
    print("sheets ->", out)


if __name__ == "__main__":
    cfg = build()
    if "--no-sheets" not in sys.argv:
        sheets(cfg)

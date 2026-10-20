# -*- coding: utf-8 -*-
"""
PROVANT v9 成片 EDL（v11）→ data/edl_v11.json（.venv_gpu python scripts/v2_preview.py V11 all；混音 python scripts/v4_audio.py v11 --mux）
用户 10-20（看过 v8 后）：
  1. 精彩片段放完整：魁札尔飞踢、贝德维尔用手化剑释放胜利誓约剑…（其余我挑）
  2. 加原片环境音
  3. intro 先不放歌：简短介绍 → 歌名 / title → 再开始放歌
  4. 中间一段 break：音乐淡出（给完整片段留时间）、设计画面、再自然地引回音乐
  5. 尾声淡出留长，音乐结束后再留一点画面和环境音
  6. 狮心王的打斗、希波吕忒多放；Richard 和阿尔托莉雅剪在一起，部分混剪
结构（成片时间 = 歌曲时间经 song_map 平移）：
  序章（无歌，原声）：Fate 召唤咏唱（FSF Whispers of Dawn）三段 + 咏唱字幕 → 标题卡 PROVANT（先 1.5 s 无歌，歌从标题卡下进来）
  歌 0–78.45：片头卡 → 伊什塔尔 → 冲田〔Alter〕→ 冲田 → 贞德 → 希波吕忒（FSF，新） → 阿尔喀德斯换气（黑边收拢）
  BREAK（歌停，原声）：slit "INTERLUDE" → 魁札尔 Ultimo Tope Patada 整段（DE002–DE038，四段连续源）+ 章节标题 + "暂停中的播放器"
  歌 78.45–：歌在 IMPERSONATOR! 清唱处回来 → 贝德维尔（银臂 → 狮子王圣枪 → 手化剑 → 胜利誓约剑释放整段）→ 武藏 → 阿拉什 → 预览 B →
             Saber Alter vs Rider → 阿尔托莉雅 × Richard 混剪 → 终幕 → 马修
  尾声：歌结束后马修再说完两句（"そういうもののために / 私たちは戦ってきたのですから"）+ 原片环境音，3 s 淡出
"""
import json, os, re, subprocess, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import edl_v10 as E10
from edl_v10 import V9, PALT, INFO, POOL, bt, item, place, shot_of, fixed_shot, slit, OLD, OLD_TR
from edl_v9 import ROOT, FF, FPS, SONG_END, B

V9.SLIT_NAME.update(hippolyta="HIPPOLYTA")
RICH = "金发红披风剑士(理查)"                                 # who → hud_v4 / palette 认作 richard
HIPPO = "希波吕忒"
EXTRA = {  # 镜头池外的源：FSF TV 第 2 弹 PV（1080，仅 Richard 打斗段；32.2 前有字幕、34.96–35.64 是吉尔伽美什，都不取）
    "PV001": dict(code="PV001", yid="EzbZP62K1-Q", n=1, **{"in": 32.50, "out": 34.90}, dur=2.40, res=1080, group="FSF", kind="战斗", grade="A",
                  who=RICH, what="霓虹屋顶上 Richard 红披风疾奔挥剑，金光迸散", usable=True, cg3d=False, gil=False, fav=True, split=False, prio=1,
                  yavg=52, ylow=10, dark=0.0, sat=0.5, motion=6.0, mood="", text="none", note="FSF TV PV2"),
    "PV003": dict(code="PV003", yid="EzbZP62K1-Q", n=3, **{"in": 2.58, "out": 3.12}, dur=0.54, res=1080, group="FSF", kind="空镜", grade="B",
                  who="none", what="夜空星点", usable=True, cg3d=False, gil=False, fav=False, split=False, prio=1,
                  yavg=12, ylow=4, dark=0.8, sat=0.3, motion=0.2, mood="", text="none", note="FSF TV PV2"),
    "PV004": dict(code="PV004", yid="EzbZP62K1-Q", n=4, **{"in": 1.96, "out": 2.55}, dur=0.59, res=1080, group="FSF", kind="特写", grade="A",
                  who="沙条绫香", what="夜街上金发眼镜少女回头", usable=True, cg3d=False, gil=False, fav=True, split=False, prio=1,
                  yavg=60, ylow=10, dark=0.0, sat=0.4, motion=0.5, mood="", text="none", note="FSF TV PV2"),
    "PV005": dict(code="PV005", yid="EzbZP62K1-Q", n=5, **{"in": 54.15, "out": 55.08}, dur=0.93, res=1080, group="FSF", kind="特写", grade="A",
                  who="沙条绫香", what="夜色中金发少女侧脸近景", usable=True, cg3d=False, gil=False, fav=True, split=False, prio=1,
                  yavg=70, ylow=10, dark=0.0, sat=0.4, motion=0.3, mood="", text="none", note="FSF TV PV2"),
    "PV002": dict(code="PV002", yid="EzbZP62K1-Q", n=2, **{"in": 35.75, "out": 36.80}, dur=1.05, res=1080, group="FSF", kind="战斗", grade="A",
                  who=RICH, what="Richard 白甲红披风跃起挥剑 → 咧嘴大笑", usable=True, cg3d=False, gil=False, fav=True, split=False, prio=1,
                  yavg=50, ylow=10, dark=0.0, sat=0.5, motion=6.0, mood="", text="none", note="FSF TV PV2"),
}
POOL.update(EXTRA)

# ---------------------------------------------------------------- 序章 / break / 尾声（成片时间，原速，画面和原声同源同步）
# (代码, 源起, 源止)；连续的源段自动合成一条原声
# 序章（用户 10-20："开头的召唤能不能换成别的，例如绫香和 richard 初识"）：
#   夜色城市（Snowfield）→ 夜街上回头的绫香 → 绫香侧脸 → 月下转身的 Richard："汝が俺のマスターか"（Whispers of Dawn 结尾，同源同步）→ 废墟竞技场里的两人远景
# (代码, 源起, 源止, 速度)
PROLOGUE = [("EP060", 228.6, 230.4, 1.0), ("PV004", 1.98, 2.52, 0.5), ("PV005", 54.18, 55.06, 0.75),
            ("EP125", 390.95, 393.54, 1.0), ("EP126", 393.54, 395.45, 1.0)]
LINE = (392.48, 394.16, "汝が俺のマスターか")               # Richard 第一句（Whisper medium：なんじがおれのマスターか）
TITLE_SRC = ("EP015", 60.90, 0.32)                           # 星空下洞口泛蓝光的山丘：标题卡底
TITLE_PRE = 1.5                                              # 标题卡先在无歌里出现 1.5 s，歌从标题卡下进来
BREAK_SONG = 78.45                                           # 歌在 77.2 起自然衰减到底；78.53 是 IMPERSONATOR! 清唱
BREAK2_SONG = 97.90                                          # 第二次停歌："And cast the first die!" 唱完（光束对撞，"逆説" 97.94 起唱之前）→ 磁带停转 → 对波打完 → "逆説歪む黎明" 回歌
# 风格化只在源片自己的剪辑点上换（用户："只用酷炫的风格化处理就行"）；不遮场景（不用 Cd / Cb）
# 用户 10-20："羽蛇神飞踢片段减少粗像素的风格化处理…人眼看的会难受" → 不用 P（版画粗像素），换美漫 / 漫画页 / riso / 干净底色 + HUD / 推镜
LOOK1 = {"DE002": dict(look="A", sfx="FWOOSH!", who="魁札尔·科亚特尔"),                   # 火球俯冲：美漫
         "DE007": dict(look="R", hud="scope", who="魁札尔·科亚特尔"),                    # 星球弧面：瞄准镜锁定落点
         "DE009": dict(look="A", sfx="WHAM!", who="魁札尔·科亚特尔"),                   # 摔角靴：美漫
         "DE011": dict(look="M", manga="impact", tone="duo", caption="ティアマト", tag="TIAMAT  //  BEAST II"),
         "DE014": dict(look="A", sfx="BOOM!", who="魁札尔·科亚特尔"),
         "DE016": dict(look="B", damp=True, poster=dict(big="QUETZALCOATL", sub="Quetzalcoatl  //  Rider  //  Ultimo Tope Patada", jp="ケツァル・コアトル", num="07")),
         "DE017": dict(look="M", manga="impact", tone="mono", tag="QUETZALCOATL  //  RIDER", damp=True),   # 源里白底 → 黑烟骤变：亮度阻尼   # 火环：漫画冲击页
         "DE023": dict(look="R", hud="hexwall", who="魁札尔·科亚特尔"),
         "DE026": dict(look="A", sfx="KRAK!"),
         "DE029": dict(look="M", manga="story"),
         "DE032": dict(look="R", hud="type", who="魁札尔·科亚特尔"),
         "DE033": dict(look="R", slam=12, burst=True, hud="lock", who="魁札尔·科亚特尔"),   # 飞踢命中：HUD 锁定 + 推镜（不做版画冲击帧）
         "DE036": dict(look="A", sfx="FWOOSH!")}
LOOK2 = {"2J048": dict(look="R", nohud=True),                                           # 白色光丝缠绕城堡：干净底色（不用版画粗像素）
         "2J049": dict(look="R", hud="lock", who="贝德维尔")}                            # 两人在强光中对冲 → 狮子王的微笑
BREAKS = [dict(song=BREAK_SONG, runs=[("DE002", "DE038", 0.083, 43.70)], look=LOOK1, chapter=True, level=-11.0, xf_in=0.1, xf_out=0.3, nat_fo=0.9),   # 魁札尔整段，一刀不剪
          # 贝德维尔（用户："贝德维尔能不能片段也持续到二人对波完"）：歌在光束对撞上停，原声把对波放完（2J047–2J049 源 53.21–70.29，一刀不剪）
          dict(song=BREAK2_SONG, runs=[("2J047", "2J049", 53.208, 70.29)], look=LOOK2, chapter=False, level=-11.0, xf_in=1.0, xf_out=0.35, nat_fo=1.5)]   # 用户："稍微将音乐淡出和渐入做的自然一些"：1 s 交叉淡出、回歌提前 0.35 s 交叉淡入
OKITA_LINES = [("OKAteito4K6", 14.35, 15.92, "冲田总司〔Alter〕", "ただひたすらに斬るのみ!（OA006 燃烧神社里的冲田〔Alter〕）"),
               ("OKAteito4K6", 18.00, 19.86, "冲田总司 / 冲田总司〔Alter〕", "やはり私はお前だ（OA008 刀锋相抵）"),
               ("OKjiend4K60", 11.92, 12.98, "冲田总司", "覚悟はいい（OJ003 冲田持刀特写）")]
MASH_TAIL = [(120.14, 121.52), (121.68, 123.42)]            # 马修："そういうもののために / 私たちは戦ってきたのですから"（接片尾三句）


def H_(code, **kw):
    return (code, dict(kw, who=kw.get("who", HIPPO)))


def R_(code, **kw):
    return (code, dict(kw, who=RICH))


SONG_BLOCKS = [
    dict(id="V01", name="INTRO", fixed=["E01_01", "E01_02", "E01_03", "E01_04", "E01_05"], kind="intro", enter="OLD"),
    *[dict(b) for b in E10.BLOCKS if b["id"] in ("V02", "V03", "V04", "V05")],
    # 59.56–77.02 希波吕忒（FSF 真 Rider，用户："希波吕忒的片段也可以多放一些"）：持枪蓄势 → 骑蓝马冲下沙坡 → 阿尔喀德斯 → 浮岩间跃马 →
    #   怒吼 → 军帯光环 → 战马金色电光 → 74.70 "Judge his master" 带电冲锋 → 白色剑痕
    dict(id="H1", name="HIPPOLYTA", t0=101, t1=131, kind="battle", hot=True, enter=slit("hippolyta"),
         items=[H_("DL040"), H_("DL041", hud="lock"), "DL042", "DL043", H_("DL044", hud="dossier"), ("DL045", dict(who="阿尔喀德斯")),
                H_("DL047"), H_("DL048"), H_("DL049"), H_("DL050", look="A", sfx="KRAK!"), H_("DL052"), H_("DL053"),
                "DL069", H_("DL070", hud="np"), H_("DL074"), H_("DL075"),
                H_("DL076", at=127, look="P", hold=0, impact=3, slam=10, burst=True, hud="type"), "DL077", H_("DL061")]),
    dict(id="V07", name="BREATH", fixed=["E06_00"], kind="breath", trim_b=BREAK_SONG),
    # ---- BREAK ----
    # 78.45–98.01 贝德维尔（用户："贝德维尔用手化剑释放胜利誓约剑也可以放完整一些"）：银臂迸光 → 狮子王圣枪 Rhongomyniad →
    #   金光中手化为剑 → 胜利誓约剑光束横扫红天、与雷光对撞（2J038–2J047 整段，v8 只放到 2J036）；96.85 "And cast the first die!" = 对撞
    dict(id="BD", name="BEDIVERE", t0=BREAK_SONG, t1=BREAK2_SONG, kind="battle", enter=slit("bedivere"),
         items=[("2J002", dict(s0=2.03, s1=3.10, hud="lock", who="贝德维尔")),
                ("2J027", dict(hud="type", who="狮子王")), "2J028", "2J029", "2J030", ("2J031", dict(hud="np", who="狮子王")),
                ("2J032", dict(s1=36.8, look="P", hold=0, nohud=True)), "2J033",
                ("2J034", dict(hud="np", who="贝德维尔")), "2J035", ("2J036", dict(s1=42.0, look="P", hold=0, nohud=True, who="贝德维尔")),
                "2J038", "2J039", ("2J040", dict(look="A", sfx="KRAK!")), ("2J041", dict(hud="type", who="贝德维尔")),
                "2J042", "2J043", "2J044", "2J045",
                ("2J046", dict(at=165, look="R", slam=12, burst=True, hud="lock", who="贝德维尔"))]),   # 对撞 → 第二次停歌
    *[dict(b, t0=BREAK2_SONG) if b["id"] == "V09" else dict(b) for b in E10.BLOCKS if b["id"] in ("V09", "V11", "V12")],   # 武藏从回歌处开始
    # 132.21–149.70 Saber Alter vs Rider（Rider 段收短 2.3 s 给阿尔托莉雅 × Richard）
    dict(id="SA", name="SABER ALTER", t0=226, t1=256, kind="climax", hot=True, enter=slit("artoria_alter"),
         items=[("RV013", dict(look="G")), ("RV014", dict(hud="lock")), ("RV015", dict(s1=19.33)), ("RV017", dict(look="M", manga="impact")),
                ("RV019", dict(look="A", sfx="CLANG!")), ("RV021", dict(take=1.0)),
                ("RV094", dict(at=243, s0=125.32, s1=126.65, look="P", hold=0, impact=3, slam=10, burst=True, hud="lock")),
                ("RV097", dict(tail=2.2, hud="np")),
                ("RV098", dict(at=249, look="P", hold=0, impact=3, slam=10, burst=True, hud="type")), "RV099",
                ("RV101", dict(look="A", sfx="FWOOSH!")), ("RV102", dict(s0=137.67, s1=138.42, look="P", hold=0, nohud=True)), ("RV103", dict(take=1.3))]),
    # 149.70–163.68 阿尔托莉雅 × Richard 混剪（用户："richard 可以和阿尔托莉雅剪在一起，部分可以用混剪"；"狮心王的打斗片段…多放一些"）
    #   两个持圣剑的王交替：背影 / 坏笑 → 仰望（灵基卡）/ 夜里回望 → 飞跃 / 屋顶疾奔挥剑 → 云海飞行 / 跃起斩击 / 夜空挥剑 → 面部 / 侧脸 → 持剑飞掠
    dict(id="AR", name="ARTORIA x RICHARD", t0=256, t1=280, kind="climax", hot=True, enter=slit("artoria"),
         items=[("PM066", dict(take=1.2, hud="type")), R_("FS024", hud="type", enter=dict(kind="tape", key="richard")),
                ("PM071", dict(w=0.8, look="G")), R_("EP125", s0=390.95, take=1.6),
                ("PM072", dict(take=0.85)), R_("PV001", hud="lock"),
                ("PM075", dict(take=1.2, look="A", sfx="FWOOSH!")), R_("PV002"), R_("FS026"),
                ("PM077", dict(hud="np")), R_("FS042", s0=62.35, hud="np"), "PM078"]),
    dict(id="V16", name="FINALE", fixed=["E11_39"], kind="climax", hot=True, enter="cut"),
]



def native_shot(code, s0, s1, t, blk, idx, opts=None):
    r = POOL[code]
    it = dict(code=code, r=r, s0=s0, s1=s1, lo=r["in"], hi=r["out"], o=dict(opts or {}), a=t, b=t + (s1 - s0), speed=1.0)
    sh = shot_of(it, blk, idx)
    if (opts or {}).get("damp"):
        sh["damp"] = True
    return sh


def run_codes(c0, c1):
    p, n0, n1 = c0[:2], int(c0[2:]), int(c1[2:])
    return [p + "%03d" % n for n in range(n0, n1 + 1) if p + "%03d" % n in POOL]


def tape_for(tr):
    if tr["kind"] == "slit" and "tape" not in tr:
        key = {v: k for k, v in V9.SLIT_NAME.items()}.get(tr["text"])
        tr.setdefault("tape_col", list(PALT.SIG[key][0]) if key else list(PALT.SIG["lion_king"][0]))
        tr["tape"] = ["CAUTION", tr["text"]] + [re.sub(r"\s+", " ", x).strip() for x in tr.get("sub", "").split("/") if x.strip()]
    return tr


PP = {"slit": (12, 18), "glitch_slam": (4, 8), "glitch": (5, 6), "whip": (6, 6), "hex": (9, 9), "tape": (10, 24), "gcut": (2, 3)}


def song_part():
    """歌曲部分：在歌曲时间里排（沿用 v10 的块排时 / 转场规则），返回 (shots, trans, mus)。"""
    V9._NM["n"] = 0
    shots, segs_of, bounds, mus = [], [], [], None
    for blk in SONG_BLOCKS:
        start = len(shots)
        if "fixed" in blk:
            for ref in blk["fixed"]:
                sh = fixed_shot(ref, blk)
                if blk.get("trim_b"):
                    sh["b"] = blk["trim_b"]
                shots.append(sh); segs_of.append(blk["kind"])
        else:
            its = place(blk)
            for k, it in enumerate(its):
                en = it["o"].pop("enter", None)
                sh = shot_of(it, blk, k)
                if en:
                    sh["_enter"] = en
                shots.append(sh); segs_of.append(blk["kind"])
                if it["code"] == "MU003":
                    mus = sh
        bounds.append((start, blk))
    for k in range(1, len(shots)):
        if abs(shots[k]["a"] - shots[k - 1]["b"]) > 1e-3:
            print("  ! 时间不连续 %s %.3f → %s %.3f" % (shots[k - 1]["id"], shots[k - 1]["b"], shots[k]["id"], shots[k]["a"]))
    auto = {round(t["t"], 3): t for t in V9.assign_trans(shots, segs_of)}
    fixed_ranges = [(shots[s]["a"], shots[s + len(b["fixed"]) - 1]["b"]) for s, b in bounds if "fixed" in b]
    trans = []
    for t in OLD_TR:
        if any(a + 1e-3 < t["t"] < b - 1e-3 for a, b in fixed_ranges):
            trans.append(dict(t, kind="glitch", pre=5, post=6) if t["kind"] == "panel_slap" else dict(t))
    for s, blk in bounds:
        t = shots[s]["a"]
        en = blk.get("enter")
        old = next((x for x in OLD_TR if abs(x["t"] - t) < 1e-3), None)
        if en == "OLD" or en is None:
            if old:
                trans.append(dict(old))
            elif en is None and round(t, 3) in auto:
                trans.append(auto[round(t, 3)])
            continue
        if en == "cut" or en.get("kind") == "cut":
            continue
        pre, post = PP[en["kind"]]
        tr = dict(en, t=t, pre=pre, post=post)
        if en["kind"] == "tape":
            key = tr.pop("key"); inf = INFO[key]
            tr["tape"] = ["CAUTION", inf["en"], "CLASS " + inf["cls"], inf.get("jp", "")]; tr["tape_col"] = list(PALT.SIG[key][0])
        trans.append(tr)
    for sh in shots:                                        # 块内单独指定的转场（混剪里 Richard 第一次出场的警戒线）
        en = sh.pop("_enter", None)
        if en:
            key = en["key"]; inf = INFO[key]
            trans = [x for x in trans if abs(x["t"] - sh["a"]) > 1e-3]
            trans.append(dict(kind="tape", t=sh["a"], pre=10, post=24, tape=["CAUTION", inf["en"], "CLASS " + inf["cls"], inf.get("jp", "")],
                              tape_col=list(PALT.SIG[key][0])))
    trans = [tape_for(x) for x in sorted(trans, key=lambda x: x["t"])]
    return shots, trans, mus


def build():
    song_shots, song_trans, mus = song_part()
    # ---- 序章（绫香与 Richard 初识）
    P = dict(id="P0", name="PROLOGUE", kind="intro")
    pro, t, caps, native = [], 0.0, [], []
    for k, (code, s0, s1, sp) in enumerate(PROLOGUE):
        r = POOL[code]
        it = dict(code=code, r=r, s0=s0, s1=s1, lo=r["in"], hi=r["out"], o={}, a=t, b=t + (s1 - s0) / sp, speed=sp)
        pro.append(shot_of(it, P, k))
        if code == "EP125":
            t_r1, s_r1 = t, s0
        t += (s1 - s0) / sp
    P_END = t
    f_line = lambda x: t_r1 + (x - s_r1)                  # EP125 / EP126 原速同步
    caps.append(dict(t0=round(f_line(LINE[0]) - 0.05, 3), t1=round(f_line(LINE[1]) + 0.6, 3), text=LINE[2], style="serif"))
    native.append(dict(t0=0.0, t1=round(t_r1 + 0.4, 3), yid="11vAh33_2eI", src=99.0, level=-30.0, fi=0.6, fo=0.5, note="序章夜风（和片尾同一段环境声，首尾呼应）"))
    native.append(dict(t0=round(t_r1 - 0.5, 3), t1=round(P_END, 3), yid="EpLKseoUV6Q", src=round(s_r1 - 0.5, 3), level=-15.0, fi=0.4, fo=0.08,
                       note="Richard 初见原声：汝が俺のマスターか"))
    SONG0 = P_END + TITLE_PRE
    # ---- 歌段 + break（成片时间）
    segs, brk, binfo, f, prev = [], [], [], SONG0, 0.0
    for bi, B_ in enumerate(BREAKS):
        segs.append((f, prev, B_["song"]))
        f += B_["song"] - prev
        F_b0 = f
        blk = dict(id="BR" if bi == 0 else "BS", name="INTERLUDE" if bi == 0 else "BEDIVERE II", kind="battle")
        k = 0
        for c0, c1, s0, s1 in B_["runs"]:
            t_run = f
            for code in run_codes(c0, c1):
                r = POOL[code]
                a_, b_ = max(r["in"], s0), min(r["out"], s1)
                brk.append(native_shot(code, a_, b_, f, blk, k, B_["look"].get(code, {}))); k += 1
                f += b_ - a_
            xi = B_["xf_in"] if t_run == F_b0 else 0.0
            native.append(dict(t0=round(t_run - xi, 3), t1=round(f, 3), yid=POOL[c0]["yid"], src=round(s0 - xi, 3), level=B_["level"], fi=max(0.03, xi),
                               fo=B_["nat_fo"], note="break %d 原声（一刀不剪）" % (bi + 1)))
        binfo.append(dict(B_, F0=F_b0, F1=f))
        prev = B_["song"]
    segs.append((f, prev, SONG_END))

    def s2f(x, end=False):
        for i, (f0, a, b) in enumerate(segs):
            if (a - 1e-6 <= x < b - 1e-6) or (end and abs(x - b) <= 1e-6) or (i == len(segs) - 1 and x >= a - 1e-6):
                return f0 + (x - a)
        raise ValueError(x)
    song_map = [[round(f0, 4), a, b] for f0, a, b in segs]
    for sh in song_shots:
        sh["sa"], sh["sb"] = sh["a"], sh["b"]
        sh["a"], sh["b"] = round(s2f(sh["a"]), 4), round(s2f(sh["b"], end=True), 4)
    for tr in song_trans:
        tr["t"] = round(s2f(tr["t"]), 4)
    # ---- 标题卡（歌从它下面进来）
    TI = dict(id="TI", name="TITLE", kind="intro")
    e0 = next(s for s in song_shots if s["id"] == "E01_01")
    code, s0, sp = TITLE_SRC
    r = POOL[code]
    it = dict(code=code, r=r, s0=s0, s1=s0 + sp * (e0["a"] - P_END), lo=r["in"], hi=r["out"], o=dict(look="T"), a=P_END, b=e0["a"], speed=sp)
    title = shot_of(it, TI, 0)
    title["title_spec"] = dict(kicker="FATE SERIES  //  AMV", title="PROVANT", sub="SawanoHiroyuki[nZk]:Jean-Ken Johnny", exit=False)
    native.append(dict(t0=round(P_END, 3), t1=round(SONG0 + 1.0, 3), yid=r["yid"], src=s0, level=-24.0, fi=0.05, fo=0.9, note="标题卡前的夜风"))
    # ---- 台词（成片时间）
    dialog = [dict(d, t=round(s2f(d["t"]), 3)) for d in V9.DIALOG if d["yid"] != "MUkengo4K60"]
    t_song = mus["sa"] + (6.976 - mus["src_a"]) / mus["speed"]
    assert 100.95 < t_song and t_song + 0.9 < 108.2, "武藏台词落进唱段"
    dialog.append(dict(t=round(s2f(t_song), 3), yid="MUkengo4K60", s0=6.96, s1=7.80, who="宫本武藏", note="MU003 同源画面同步"))
    for yid, s0_, s1_, who_, note_ in OKITA_LINES:          # 按同源画面同步：找源时间落在哪个镜头里
        sh = next(x for x in song_shots if x["yid"] == yid and x["src_a"] - 0.02 <= s0_ <= x["src_a"] + (x["sb"] - x["sa"]) * x["speed"] + 0.02)
        dialog.append(dict(t=round(sh["a"] + (s0_ - sh["src_a"]) / sh["speed"], 3), yid=yid, s0=s0_, s1=s1_, who=who_, note=note_))
    F_SONG_END = s2f(SONG_END, end=True)
    t4 = s2f(170.40)
    dialog.append(dict(t=round(t4, 3), yid="Cf9uMIffDTQ", s0=MASH_TAIL[0][0], s1=MASH_TAIL[0][1], who="玛修", note="歌结束后：そういうもののために"))
    t5 = t4 + (MASH_TAIL[0][1] - MASH_TAIL[0][0]) + 0.22
    dialog.append(dict(t=round(t5, 3), yid="Cf9uMIffDTQ", s0=MASH_TAIL[1][0], s1=MASH_TAIL[1][1], who="玛修", note="私たちは戦ってきたのですから"))
    T1 = round(t5 + (MASH_TAIL[1][1] - MASH_TAIL[1][0]) + 1.2 + 3.0, 3)
    dialog.sort(key=lambda d: d["t"])
    # ---- 尾声：马修 Cs 延长到片尾（源 96.1–105.7 摊开）
    OU = dict(id="E12", name="MASH", kind="outro")
    a_cs = s2f(165.5815)
    r = POOL["11065"]
    it = dict(code="11065", r=r, s0=96.10, s1=105.70, lo=r["in"], hi=r["out"], o=dict(look="Cs"), a=a_cs, b=T1, speed=(105.70 - 96.10) / (T1 - a_cs))
    cs = shot_of(it, OU, 0); cs["id"] = "E12_00"; cs["seg"] = "E12"
    native.append(dict(t0=round(s2f(167.5), 3), t1=T1, yid=r["yid"], src=99.0, level=-27.0, fi=2.0, fo=3.0, note="尾声环境音（马修那场的原声）"))
    song_shots = [s for s in song_shots if s["id"] != "E12_00"]
    shots = sorted(pro + [title] + song_shots + brk + [cs], key=lambda s: s["a"])
    for kx in range(1, len(shots)):
        if abs(shots[kx]["a"] - shots[kx - 1]["b"]) > 2e-3:
            print("  ! 成片不连续 %s %.3f → %s %.3f" % (shots[kx - 1]["id"], shots[kx - 1]["b"], shots[kx]["id"], shots[kx]["a"]))
    V9.assign_hud([s for s in shots if s["look"] in ("R", "P") and not s.get("nohud")])
    # ---- 转场（成片时间）
    trans = list(song_trans)
    for sh in pro[1:]:
        trans.append(dict(t=sh["a"], kind="gcut", pre=2, post=3)) if sh["code"] not in ("EP126",) else None   # EP125 → EP126 是源里的原生剪辑
    trans.append(dict(t=round(P_END, 4), kind="glitch", pre=5, post=6))
    b1 = binfo[0]
    trans.append(tape_for(dict(t=round(b1["F0"], 4), kind="slit", pre=12, post=18, text="INTERLUDE", sub="第七特異点  /  絶対魔獣戦線バビロニア",
                               tape_col=list(PALT.SIG["quetzalcoatl"][0]))))
    trans[-1]["tape"] = ["CAUTION", "INTERLUDE", "SOURCE AUDIO", "QUETZALCOATL"]
    old = next((x for x in OLD_TR if abs(x["t"] - 165.5815) < 1e-3), None)
    if old:
        trans.append(dict(old, t=round(a_cs, 4)))
    trans = sorted(trans, key=lambda x: x["t"])
    print("  转场:", dict(Counter(x["kind"] for x in trans)), " slit:", [x["text"] for x in trans if x["kind"] == "slit"])
    # ---- 歌词 / glitch / 爆点 / 字幕
    song_view = [dict(s, a=s["sa"], b=s["sb"]) for s in shots if "sa" in s]
    lyr = V9.lyric_map(song_view)
    glitch = [(round(s2f(a), 3), round(s2f(b, end=True), 3), x, y) for a, b, x, y in E10.GLITCH + V9.beat_glitch()
              if not any(abs(a - B_["song"]) < 0.3 for B_ in BREAKS)]
    glitch += [(round(P_END - 0.2, 3), round(P_END, 3), 0.1, 0.5), (round(P_END, 3), round(P_END + 0.12, 3), 0.8, 0.0)]
    for bi, b in enumerate(binfo):
        glitch += [(round(b["F0"] - 0.30, 3), round(b["F0"], 3), 0.1, 0.55), (round(b["F0"], 3), round(b["F0"] + 0.14, 3), 0.9, 0.0),
                   (round(b["F1"] - 0.35, 3), round(b["F1"], 3), 0.12, 0.65), (round(b["F1"], 3), round(b["F1"] + 0.14, 3), 1.0, 0.0)]
        caps.append(dict(t0=round(b["F0"] + 0.3, 3), t1=round(b["F1"] + 1.3, 3), style="player", title="PROVANT  ―  SawanoHiroyuki[nZk]",
                         resume=round(b["F1"], 3), song_t=b["song"], song_len=SONG_END, label_pause="PAUSED  ·  SOURCE AUDIO", label_play="NOW PLAYING"))
        if b.get("chapter"):
            caps.append(dict(t0=round(b["F0"] + 0.9, 3), t1=round(b["F0"] + 4.8, 3), style="chapter", kicker="INTERLUDE  //  QUETZALCOATL  ·  RIDER",
                             text="ULTIMO TOPE PATADA", sub="ケツァル・コアトル  ―  絶対魔獣戦線バビロニア"))
    punch = [round(s2f(x), 3) for x in E10.PUNCH]
    cfg = dict(T0=0.0, T1=T1, out="PROVANT_AMV_v9_video.mp4", shots=shots, trans=trans, lyrics=lyr, glitch=glitch, punch=punch,
               fade=[(round(T1 - 3.0, 3), T1)], dialog=dialog, hud_v=5, song_map=song_map, native_audio=native, captions=caps,
               song_xf=[dict(fo=BREAKS[0]["xf_in"] if BREAKS[0]["xf_in"] > 0.5 else 0.3)] +
                       [dict(pre=BREAKS[k]["xf_out"], fo=(BREAKS[k + 1]["xf_in"] if k + 1 < len(BREAKS) else 0.0)) for k in range(len(BREAKS))])
    fw = V9.flash_check(json.loads(json.dumps(shots)))
    fw = [w for w in fw if not w[1].startswith("slide")]
    dd = V9.dupes(shots, {s["code"].split(">")[0] for s in shots})
    for s in shots:
        s.pop("_risky", None)
    json.dump(cfg, open(os.path.join(ROOT, "data", "edl_v11.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # ---- 统计 + 人看的表
    n = len(shots)
    lk = Counter(s["look"] + ("-" if s.get("nohud") else "") for s in shots)
    gpt = Counter()
    for s in shots:
        gpt[s["group"]] += s["b"] - s["a"]
    Ls = ["# EDL v11（自动生成：scripts/edl_v11.py）", "",
          "成片 %.2f s；序章 0–%.2f（无歌）→ 标题 → 歌 %.2f 起；break %s；歌结束 %.2f；淡出到 %.2f" % (
              T1, P_END, SONG0, "、".join("%.2f–%.2f（%.1f s）" % (b["F0"], b["F1"], b["F1"] - b["F0"]) for b in binfo), F_SONG_END, T1),
          "镜头 %d；风格 %s；转场 %s；系列时长 %s" % (n, dict(lk), dict(Counter(x["kind"] for x in trans)), {k_: round(v, 1) for k_, v in gpt.items()}), ""]
    cur = None
    for s in shots:
        if s["block"] != cur:
            cur = s["block"]; Ls += ["", "## %s" % cur]
        Ls.append("- `%s` %7.3f–%7.3f %-3s%s %-10s x%.2f %s/%s %s — %s（%s）" % (
            s["id"], s["a"], s["b"], s["look"], "-" if s.get("nohud") else (s.get("hud", "") and ":" + s["hud"]), s["code"], s.get("speed", 1),
            s["group"], s["res"], s["kind"], s["what"][:40], s["who"][:16]))
    Ls += ["", "## 台词"] + ["- %.2f %s %s" % (d["t"], d["who"], d.get("note", "")) for d in dialog]
    Ls += ["", "## 原声"] + ["- %.2f–%.2f %s @%.2f %s" % (x["t0"], x["t1"], x["yid"], x["src"], x["note"]) for x in native]
    Ls += ["", "## 预检提示"] + ["- " + str(w) for w in fw] + ["- 画面重复：" + str(dd)]
    open(os.path.join(ROOT, "EDL_v11.md"), "w", encoding="utf-8").write("\n".join(Ls))
    print(Ls[2]); print(Ls[3])
    return cfg


def sheets(cfg):
    from cards import src_path
    from PIL import Image, ImageDraw, ImageFont
    out = os.path.join(ROOT, "ae", "qa", "v11"); os.makedirs(out, exist_ok=True)
    fnt = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 15)
    by = {}
    for s in cfg["shots"]:
        by.setdefault(s["block"], []).append(s)
    tr_at = {round(t["t"], 3): t["kind"] for t in cfg["trans"]}
    for k, (blk, ss) in enumerate(by.items()):
        cols = 6; rows = (len(ss) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 320, rows * 210), (16, 16, 18)); d = ImageDraw.Draw(sheet)
        for j, s in enumerate(ss):
            t = s["src_a"] + (s["b"] - s["a"]) * s.get("speed", 1) * 0.5
            p = subprocess.run([FF, "-v", "error", "-ss", "%.3f" % min(t, s["hi"]), "-i", src_path(s["yid"]), "-frames:v", "1",
                                "-vf", "scale=320:180", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
            if len(p.stdout) >= 320 * 180 * 3:
                sheet.paste(Image.frombytes("RGB", (320, 180), p.stdout[:320 * 180 * 3]), ((j % cols) * 320, (j // cols) * 210))
            d.text(((j % cols) * 320 + 4, (j // cols) * 210 + 181), "%s %s%s %s %.2f %s" % (s["id"][-2:], s["look"], "-" if s.get("nohud") else "",
                                                                                       s["code"], s["a"], tr_at.get(round(s["a"], 3), "|")), font=fnt, fill=(255, 220, 90))
        sheet.save(os.path.join(out, "%02d_%s.jpg" % (k, re.sub(r"\W", "_", blk))), quality=85)
    print("sheets ->", out)


if __name__ == "__main__":
    cfg = build()
    if "--no-sheets" not in sys.argv:
        sheets(cfg)

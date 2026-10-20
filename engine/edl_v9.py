# -*- coding: utf-8 -*-
"""
PROVANT v7 成片 EDL（v9）→ data/edl_v9.json（python scripts/v2_preview.py V9 all）
v9：双人对打漫画页 tone=duo（两人各一色 riso 套印）；E05d_03（1:11 Saber Alter）改 R、不做环境蒙版；
强拍加短 glitch 脉冲、1 拍一切的硬切全改 gcut（"glitch 效果可以多加一点"）。
（以下为 v8 说明）
PROVANT v6 成片 EDL（v8）→ data/edl_v8.json
v8：漫画页默认单色（tone=mono，人物配色双色调；黑白只留 PM078 / MU001 两页）；冲田 / 冲田〔Alter〕/ 武藏 / 贞德不用美漫；
美漫说明框拆成名字 + 职阶小标。
（以下为 v7 说明）
PROVANT v5 成片 EDL（v7）→ data/edl_v7.json
v7：漫画页默认全彩（tone=color，黑白只留 2 页）、漫画页频率减半、美漫加量；HUD 去掉 circle（"法阵…很像是两个图层"）；
E02_14 换 PM107（CF080 源只有 0.96 s，拉长后跑进淡出黑场 = 0:27 黑画面）；E03 冲田〔Alter〕海报改 Cd（JUST LIKE ALL THE REST 现在唱到 38.2 s，
海报上不叠歌词会吞掉 ALL）；台词只放在不唱的段落（31–38 / 100.5–102.7 实际在唱）。
（以下为 v6 说明）
PROVANT v4 成片 EDL（v6）→ data/edl_v6.json
在 v5 上按用户看过 v3 的 7+1 条改：HUD v5（hud=模板，精密 UI + 编排动效 + 角色配色）、A 美漫、G 灵基卡（每个英灵一套框）、
slit 画幅收窄强调转场、台词只配给画面上的说话人（E01 贞德同步口型、片尾马修三句）、片尾 Cs 柔边、慢唱句逐词。
在 v4 上按用户 10-19 的 9 条改：
  · 第 2 条 HUD 覆盖太长 → 引入漫画分镜 M（band / impact / story / slant，G11 规则）+ riso 角色海报 B（只放在无人声的空档）
  · 第 3 条 冲田总司 / 冲田〔Alter〕/ 宫本武藏 CM 素材全部用上：E03 = 帝都聖杯奇譚块，E07 尾 = 冲田块，E08 = 武藏块，E10 / E11 补镜
  · 第 5 条 EVA 蜂巢转场 hex（约 15 个）
  · 第 8 条 女性台词放在人声空档（cfg["dialog"] 只存时间与源，文字在 dialog_candidates）
  · 歌词文字不进 EDL：漫画竖排框用 caption_lyr = 歌词行号，由引擎从用户包里取
（以下为 v4 说明）

用户 10-18 定的方向（预览 A / B 认可后："可以"）：
  · 每镜一种风格层：R = Raf0x00 战术 HUD（hud_v2）/ Cd = causasui 暗态星座 / Cb = causasui 亮态档案 / P = hectoroz 版画（+HUD）
  · 转场库：lock_zoom（HUD 锁定接力）、panel_slap（档案拍贴）、glitch / gcut / glitch_slam、whip（甩镜）、grid（网格错位）；
    硬切 ≤ 30%，且只落在 1 拍一切的密集段
  · 歌词：英文 hook / 高潮句逐词（日文按短语）炸出；高潮（hot）加乱码解码、回声、每拍脉冲、荧光条、碎裂退场，峰值词加词墙
  · 高潮：E10 尾 → E11 头 = 预览 B 原样（蓄力 / 换气 / 128.128 爆点）；E05、E07 段尾、E11 内 142.1 / 145.6 / 163.7 做小爆点
  · 素材：多用 FSN 剧场版（HF）、FSF、FGO（"可以的话你可以多用一些"）；2K+ 优先；无 3D；吉尔伽美什不当主角；
    卡尔纳 vs 齐格飞允许裁掉水印（pool_v2 里 crop=1.05）
节奏骨架沿用 v1/v3 的切点（已验证的卡点），E10 尾 / E11 头按预览 B 改成每拍一切，E11 中段在 142.1 s 加一刀。
输出：data/edl_v9.json、EDL_v9.md、ae/qa/v9/<段>.jpg
"""
import json, math, os, re, subprocess, sys
from collections import Counter
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
ROOT = r"D:\A\provant"
FF = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
FPS, W, H = 60, 2560, 1440
SONG_END = 171.017
G = json.load(open(os.path.join(ROOT, "data", "pulse_grid_adaptive.json")))
B = G["beat_times_s"]
DOWN = G["downbeat_times_s"]
EN = json.load(open(os.path.join(ROOT, "data", "beat_energy.json"), encoding="utf-8"))
LYR = json.load(open(os.path.join(ROOT, "research", "lyrics_user", "ja_en", "PROVANT_逐句时间数据.json"), encoding="utf-8"))["rows"]
POOL = {r["code"]: r for r in json.load(open(os.path.join(ROOT, "data", "shot_pool_v2.json"), encoding="utf-8"))}
AUD = json.load(open(os.path.join(ROOT, "data", "shots", "cg3d_audit.json"), encoding="utf-8"))
import pool_v2 as PV


def bt(b):
    if b == "END":
        return SONG_END
    return 0.0 if b == 0 else B[b]


def rng(a, b, s):
    return list(range(a, b, s))


def impact(b):
    return EN[b]["impact"] if 0 <= b < len(EN) else 0.0


# ---------------------------------------------------------------- 角色 → HUD 文案
CHAR = [  # (匹配, 标题, tags, 星座标签)
    ("冲田总司〔Alter〕|冲田〔Alter〕", "OKITA SOUJI [ALTER]", ["CLASS  ALTER EGO", "NP     ZEKKEN MUKYUU SANDAN", "STATUS MAJIN"],
     ["ALTER", "MAJIN", "TEITO", "BLACK", "SAKURA", "MOON", "VOID", "BLADE"]),
    ("冲田总司", "OKITA SOUJI", ["CLASS  SABER", "NP     MUMYOU SANDANZUKI", "UNIT   SHINSENGUMI"],
     ["SHINSENGUMI", "MUMYOU", "SANDAN", "SAKURA", "KYOTO", "FIRST UNIT", "SWORD", "ASURA"]),
    ("宫本武藏", "MIYAMOTO MUSASHI", ["CLASS  SABER", "NP     ROKUDOU GORIN KURIKARA", "STYLE  NITEN ICHIRYUU"],
     ["NITEN", "ICHIRYUU", "KURIKARA", "GORIN", "EMPTINESS", "SHIMOSA", "WANDERER", "TWO SWORDS"]),
    ("Saber Alter", "SABER ALTER", ["CLASS  SABER [ALTER]", "PARAM  STR A / MGI A++", "NP     EXCALIBUR MORGAN"],
     ["ALTER", "MORGAN", "BLACK", "VORTIGERN", "OATH", "KING", "CORRUPT", "GRAIL"]),
    ("阿尔托莉雅|Saber$|^Saber|着甲Saber", "ARTORIA PENDRAGON", ["CLASS  SABER", "PARAM  STR B / AGI B", "NP     EXCALIBUR"],
     ["SABER", "KING", "AVALON", "CAMLANN", "OATH", "SWORD", "LAKE", "BRITAIN"]),
    ("贞德", "JEANNE D'ARC", ["CLASS  RULER", "PARAM  MGI A / LCK C", "NP     LUMINOSITE ETERNELLE"],
     ["RULER", "LA PUCELLE", "BANNER", "ORLEANS", "1412", "FAITH", "SAINT", "FLAME"]),
    ("库·丘林", "CU CHULAINN", ["CLASS  LANCER", "NP     GAE BOLG", "RANGE  2-4 / 1"], ["LANCER", "ULSTER", "HOUND", "GAE BOLG", "CURSE", "SPEAR", "RUNE", "HEART"]),
    ("Rider", "MEDUSA", ["CLASS  RIDER", "NP     BELLEROPHON", "EYES   CYBELE"], ["RIDER", "GORGON", "CYBELE", "PEGASUS", "BLOOD", "CHAIN", "SHRINE", "BREAKER"]),
    ("Berserker", "HERACLES", ["CLASS  BERSERKER", "NP     GOD HAND x12", "STATUS MADNESS"], ["BERSERKER", "TWELVE", "TRIALS", "GOD HAND", "RAGE", "GIANT", "HYDRA", "STONE"]),
    ("真Assassin", "HASSAN", ["CLASS  ASSASSIN", "NP     ZABANIYA", "TARGET HEART"], ["ASSASSIN", "ZABANIYA", "HEART", "SHADOW", "CURSE", "HAND", "MASK", "SKULL"]),
    ("卫宫士郎|士郎", "EMIYA SHIROU", ["CLASS  MASTER", "CIRCUIT 27", "TRACE  ON"], ["TRACE", "ON", "CIRCUIT", "BLADE", "ARM", "IDEAL", "HERO", "FAKER"]),
    ("Archer", "EMIYA [ARCHER]", ["CLASS  ARCHER", "NP     UNLIMITED BLADE WORKS", "SHIELD RHO AIAS"], ["ARCHER", "BLADE", "STEEL", "FIRE", "RHO AIAS", "UNKNOWN", "IDEAL", "COUNTER"]),
    ("远坂凛|凛", "TOHSAKA RIN", ["CLASS  MASTER", "ELEMENT AVERAGE ONE", "GEM    x7"], ["GEM", "JEWEL", "TOHSAKA", "RED", "STAR", "MAGECRAFT", "SWORD", "ZELRETCH"]),
    ("樱", "MATOU SAKURA", ["CLASS  MASTER", "STATUS CORRUPTED", "SHADOW IMAGINARY"], ["SHADOW", "IMAGINARY", "WORM", "GRAIL", "RED", "VOID", "SAKURA", "BLOOD"]),
    ("玛修", "MASH KYRIELIGHT", ["CLASS  SHIELDER", "NP     LORD CAMELOT", "LINK   CHALDEA"], ["SHIELDER", "CHALDEA", "SHIELD", "CAMELOT", "SENPAI", "FUTURE", "HOPE", "LIGHT"]),
    ("伊莉雅", "ILLYASVIEL", ["CLASS  MASTER", "HOMUNCULUS EINZBERN", "GRAIL  VESSEL"], ["EINZBERN", "WINTER", "SNOW", "VESSEL", "GRAIL", "SISTER", "WHITE", "CASTLE"]),
    ("齐格", "SIEG", ["CLASS  SABER [SUB]", "HOMUNCULUS", "NP     BALMUNG"], ["HOMUNCULUS", "DRAGON", "BALMUNG", "HEART", "SIEG", "WISH", "FREE", "LIFE"]),
    ("莫德雷德", "MORDRED", ["CLASS  SABER", "NP     CLARENT BLOOD ARTHUR", "STATUS REBEL"], ["MORDRED", "REBEL", "CLARENT", "RED", "KNIGHT", "FATHER", "CROWN", "THUNDER"]),
    ("塞米拉米斯", "SEMIRAMIS", ["CLASS  ASSASSIN", "NP     HANGING GARDENS", "POISON A"], ["ASSYRIA", "GARDEN", "POISON", "QUEEN", "DOVE", "THRONE", "SKY", "CHAIN"]),
    ("迦尔纳", "KARNA", ["CLASS  LANCER", "NP     VASAVI SHAKTI", "ARMOR  KAVACHA"], ["SUN", "KAVACHA", "VASAVI", "LANCE", "FLAME", "HONOR", "KARNA", "DIVINE"]),
    ("天草", "AMAKUSA SHIROU", ["CLASS  RULER", "NP     TWIN ARM", "WISH   SALVATION"], ["RULER", "SALVATION", "GRAIL", "PRAYER", "TWIN ARM", "BIG CRUNCH", "WISH", "SAINT"]),
    ("恩奇都", "ENKIDU", ["CLASS  LANCER", "NP     ENUMA ELISH", "FORM   CLAY"], ["CLAY", "CHAIN", "ENUMA", "ELISH", "FRIEND", "FOREST", "GOD", "WEAPON"]),
    ("提亚马特", "TIAMAT", ["CLASS  BEAST II", "NP     NEGA GENESIS", "STATUS PRIMORDIAL"], ["BEAST II", "MOTHER", "SEA", "GENESIS", "HORN", "LAHMU", "CHAOS", "TIDE"]),
    ("贝德维尔", "BEDIVERE", ["CLASS  SABER", "NP     SWITCH ON AIRGETLAM", "ARM    SILVER"], ["SILVER", "ARM", "KNIGHT", "RETURN", "CAMELOT", "OATH", "LAKE", "KING"]),
    ("狮子王", "LION KING", ["CLASS  LANCER", "NP     RHONGOMYNIAD", "STATUS GODDESS"], ["RHONGOMYNIAD", "GODDESS", "LION", "SPEAR", "HOLY CITY", "KING", "STORM", "LIGHT"]),
    ("伊什塔尔", "ISHTAR", ["CLASS  ARCHER", "NP     AN GALTA KIGALSHE", "STATUS GODDESS"], ["VENUS", "GODDESS", "GEM", "BOW", "HEAVEN", "MAANNA", "STAR", "GOLD"]),
    ("尼禄", "NERO CLAUDIUS", ["CLASS  SABER", "NP     AESTUS DOMUS AUREA", "STATUS EMPEROR"], ["ROSE", "ROME", "EMPEROR", "THEATER", "RED", "GOLD", "UMU", "AESTUS"]),
    ("山之翁|骷髅面具", "KING HASSAN", ["CLASS  ASSASSIN", "NP     AZRAEL", "BELL   TOLLS"], ["AZRAEL", "BELL", "DEATH", "OLD MAN", "MOUNTAIN", "SKULL", "BLUE", "FIRE"]),
    ("伊斯坎达尔", "ISKANDAR", ["CLASS  RIDER", "NP     IONIOI HETAIROI", "ARMY   ETERNAL"], ["ARMY", "KING", "OCEANUS", "IONIOI", "CONQUEST", "HORSE", "MACEDON", "DREAM"]),
    ("阿喀琉斯", "ACHILLES", ["CLASS  RIDER", "NP     TROIAS TRAGOIDIA", "HEEL   WEAK"], []),
    ("喀戎", "CHIRON", ["CLASS  ARCHER", "NP     ANTARES SNIPE", "STATUS TEACHER"], []),
    ("阿塔兰忒", "ATALANTE", ["CLASS  ARCHER", "NP     PHOEBUS CATASTROPHE", "STATUS HUNTRESS"], []),
    ("阿斯托尔福", "ASTOLFO", ["CLASS  RIDER", "NP     HIPPOGRIFF", "STATUS FREE"], []),
    ("藤丸|黑发蓝瞳少年|男主人公|黑发少年|主人公", "FUJIMARU RITSUKA", ["CLASS  MASTER", "LINK   CHALDEA", "SEAL   x3"],
     ["MASTER", "CHALDEA", "SEAL", "HUMAN", "ORDER", "FUTURE", "SENPAI", "GRAND"]),
    ("金发持剑青年|红披风剑士|金发红披风", "RICHARD I", ["CLASS  SABER", "TRUE   RICHARD I", "NP     ALWAYS EXCALIBUR"],
     ["LIONHEART", "CRUSADE", "KING", "SWORD", "ALWAYS", "SABER", "FAKE", "SNOWFIELD"]),
    ("棕发弓兵|阿拉什", "ARASH", ["CLASS  ARCHER", "NP     STELLA", "RANGE  2500 KM"], []),
    ("红发女战士", "HIPPOLYTA", ["CLASS  RIDER", "TRUE   HIPPOLYTA", "NP     ANDROMACHE"], []),
    ("黑面罩", "ALCIDES", ["CLASS  ARCHER", "TRUE   ALCIDES", "STATUS AVENGER"], []),
    ("言峰", "KOTOMINE KIREI", ["CLASS  MASTER", "STATUS EXECUTOR", "BLACK  KEYS"], []),
    ("佐佐木", "SASAKI KOJIRO", ["CLASS  ASSASSIN", "NP     TSUBAME GAESHI", "GATE   RYUDOU"], []),
]
GEN_LAB = ["SUBJECT", "MANA", "CIRCUIT", "ORIGIN", "OATH", "GRAIL", "SEAL", "RECORD"]


def char_of(who, what=""):
    for pat, title, tags, labs in CHAR:
        if re.search(pat, who or ""):
            return title, tags, (labs or GEN_LAB)
    return "UNIDENTIFIED", ["CLASS  ------", "SIGNAL  UNSTABLE", "LINK   SEARCHING"], GEN_LAB


# 漫画竖排说明框用的日文名（竖排逐字直立绘制，所以避开长音符"ー"和括号）
JPN = [("冲田总司〔Alter〕|冲田〔Alter〕", "沖田オルタ"), ("冲田总司", "沖田総司"), ("宫本武藏", "宮本武蔵"), ("贞德", "ジャンヌ・ダルク"),
       ("Saber Alter", "黒き騎士王"), ("阿尔托莉雅|Saber$|^Saber|着甲Saber", "アルトリア"), ("玛修", "マシュ"),
       ("金发持剑青年|红披风剑士|金发红披风", "獅子心王"), ("卫宫士郎|士郎", "衛宮士郎"), ("远坂凛|凛", "遠坂凛"), ("樱", "間桐桜"),
       ("迦尔纳", "カルナ"), ("言峰", "言峰綺礼"), ("Archer", "エミヤ")]


def jp_of(who):
    for pat, name in JPN:
        if re.search(pat, re.split(r"[、,，；;/]", who or "")[0].strip()):
            return name
    return None


# ---------------------------------------------------------------- 段落简报
# pin：{槽位: "CODE" 或 dict(code=, look=, src_a=, speed=, lo=, hi=, ...)}；look 不写就按规则分
PB = {  # 预览 B 原样（124.204–132.214）
    212: dict(code="SA008", look="R", det=["face", "head"]),                 # Saber Alter 面甲金瞳直视（HF3 8K）
    213: dict(code="A2038", look="Cd"),                  # 成片去重：预览 B 的对镜贞德已在 E02 用过
    214: dict(code="KK002", look="R", det=["face", "head"]),                 # 士郎 vs 言峰：拳击大特写
    215: dict(code="SA047", look="P", hold=12, cropbox="EYES // RIDER", det=["face", "head"]),   # RV005 与 E02 的 SA006 同画面
    216: dict(code="FS043", look="R", src_a=63.22, speed=1.0, lo=63.21, hi=63.74, det=["face", "head"], title="RICHARD I"),
    217: dict(code="A2002", look="P", src_a=12.80, speed=1.0, lo=12.62, hi=15.2, hold=12, cropbox="RULER // EYES", det=["face", "head"]),
    # 爆点：Saber Alter 橙色火花 → 接住 Rider 短剑的近景（SA013 源内 29.62 起；同镜头 30.2 处有内切，只用到 30.2 前）
    218: dict(code="SA013", look="P", src_a=29.62, speed=0.0, lo=29.60, hi=30.18, det=["person", "head"], manual=[900, 200, 1700, 1300],
             boil=True, hud_rate=0.42, growth=True, tension=True, push=0.08, bars=0.11),
    219: dict(code="SA013", look="R", src_a=29.62, speed=1.0, lo=29.60, hi=30.18, det=["person", "head"], manual=[900, 200, 1700, 1300],
             impact=3, slam=12, burst=True, tear=[128.128]),
    220: dict(code="FS043", look="R", src_a=63.85, speed=1.0, lo=63.75, hi=64.66, det=["person", "head"], burst=True, title="RICHARD I"),
    221: dict(code="A2002", look="Cd", src_a=13.90, speed=1.0, lo=12.62, hi=15.2),
    222: dict(code="KK025", look="R", det=["person"], tear=[129.869], burst=True),   # 圣杯巨影前的两道剪影（HF3 4K）
    224: dict(code="A1033", look="Cb", src_a=87.62, speed=1.0, lo=87.33, hi=90.08, titles=["SIEG // HOMUNCULUS", "RULER // JEANNE D'ARC"]),
}
PA = {  # 预览 A 原样（20.9–30.0；E02 尾 + E03 头）
    35: dict(code="SA006", look="R", det=["face", "head"]),     # 成片去重：SB004 留给高潮蓄力
    38: dict(code="KK005", look="R", det=["face", "head"]),                  # 士郎怒吼侧脸
    40: dict(code="A1016", look="Cd", src_a=44.78, speed=0.8, lo=44.64, hi=46.84, keep_x=1230),
    44: dict(code="PM107", look="Cb", titles=["SHIELDER // MASH KYRIELIGHT", "CHALDEA // RECORD"]),   # CF080 源太短，拉长后进黑场   # A2045 人物太小，档案页几乎全空
    48: dict(code="KK004", look="R", det=["person"], tear=[29.211]),         # 红雾里全身互殴，士郎臂上长刀
}
SEGS = [
    dict(seg="E01", b0=0, b1=15, cuts=[0, 4, 8, 11, 13, 14], kind="intro", src=["EP", "DE", "RS", "PM", "FS", "G9"],
         pin={0: dict(code="EP003", look="R", title="SUMMONING SEQUENCE", hud="scope"),
              # 贞德说"未来を取り戻す物語だったでしょ"：画面就是她本人开口（源 85.87 s 起，同步）→ 灵基卡里说话
              1: dict(code="CF045", look="G", src_a=85.77, speed=1.0, lo=85.75, hi=88.20),
              2: dict(code="RS015", look="R", hud="scope"), 4: "L0103", 5: "L0102",
              3: dict(code="FS043", look="B", src_a=64.40, speed=0.2, lo=64.30, hi=64.66,
                      poster=dict(big="RICHARD", sub="Richard I  //  Saber  //  Lionheart", jp="獅子心王", num="01"))}),
    dict(seg="E02", b0=15, b1=48, cuts=[15, 17, 19, 21, 23, 25, 27, 28, 29, 31, 33, 35, 38, 40, 44], kind="battle",
         src=["FS", "EP", "DL", "RV", "SA", "KK"],
         pin={0: dict(code="DL019", look="R"), 1: dict(code="FS024", look="Cd"), 2: "FS045", 3: dict(code="DL052", look="Cd"),
              4: "DL076", 5: "EP044", 6: "DL041", 7: dict(code="FS026", title="RICHARD I"), 9: dict(code="RV007", look="Cd"), 10: "EP113",
              11: PA[35], 12: PA[38], 13: PA[40], 14: PA[44]}),
    dict(seg="E03", b0=48, b1=65, cuts=[48, 51, 54, 56, 58, 60, 62], kind="lyrical", src=["OA"],
         pin={0: dict(code="OA009", look="M", manga="slant", caption="帝都聖杯奇譚", tag="TEITO // HOLY GRAIL"),
              1: dict(code="OA006", look="R", det=["person", "head"]),
              2: dict(code="OA010", look="M", manga="band"),
              3: dict(code="OA013", look="Cd", speed=0.5),
              4: dict(code="OA012", look="R", det=["person"]),
              5: dict(code="OA007", look="R", det=["person"]),
              6: dict(code="OA008", look="M", manga="story", src_a=18.45, speed=0.25, lo=18.16, hi=18.976, tag="OKITA SOUJI // ALTER")}),
    dict(seg="E04", b0=65, b1=67, cuts=[65, 66], kind="punch", src=["KK"], hot=True,
         pin={0: dict(code="KK018", look="P", hold=12), 1: dict(code="KK009", look="R", burst=True)}),
    dict(seg="E05a", b0=67, b1=83, cuts=rng(67, 79, 2) + [79, 80, 81, 82], kind="battle", src=["KK", "RS", "RN", "PR"],   # 士郎 vs 言峰（HF3）
         pin={0: "KK006", 2: "KK027", 3: "KK003", 5: "KK029", 7: "PR018", 8: "PR029"}),    # 宣传片里独有：士郎背后刺出剑刃 / 凛的宝石剑
    dict(seg="E05b", b0=83, b1=99, cuts=rng(83, 95, 2) + [95, 96, 97, 98], kind="battle", src=["2J", "QI"],
         pin={0: dict(code="2J002", src_a=2.03, speed=0.9, lo=1.9747, hi=3.10), 1: "2J006",
              2: "2J013", 3: "2J030", 4: "QI012", 5: "QI015", 7: "2J032"}),     # 2J002：源 3.144 s 有一帧闪黑（成片 QC 50.07 s）
    dict(seg="E05c", b0=99, b1=115, cuts=rng(99, 111, 2) + [111, 112, 113, 114], kind="battle", src=["DE", "YU", "O1"],
         pin={0: "DE010", 1: "DE028", 2: "DE031", 3: "YU027", 4: "O1009", 5: "YU103", 6: "DE038", 7: "YU114", 9: "O1004"}),
    dict(seg="E05d", b0=115, b1=131, cuts=rng(115, 127, 2) + [127, 128, 129, 130], kind="battle", src=["RV", "SA", "DL", "FS"], hot=True,
         pin={0: "SA016", 1: "RV015", 2: "SA002", 3: dict(code="RV013", look="R"), 4: "SA030",     # 1:11 Saber Alter：不加环境蒙版 5: "RV098", 6: dict(code="KK028", look="P", impact=3, slam=10, burst=True),
              7: "RV101", 9: dict(code="RV083", look="P", hold=12)}),
    dict(seg="E06", b0=131, b1=135, cuts=[131], kind="breath", src=["DL"],
         pin={0: dict(code="DL068", look="P", speed=0.0, boil=True, tension=True, push=0.06, bars=0.10, growth=True, hud_rate=0.3, title="ALCIDES")}),
    dict(seg="E07", b0=135, b1=167, cuts=[135, 137, 139, 140, 141, 143, 145, 147, 148, 149, 151, 153, 155, 156, 157, 159, 161, 162, 163,
                                          164, 165, 166], kind="battle", src=["A2", "CF", "PM", "7R", "AK", "AS", "2J", "RV"],
         pin={0: dict(code="A2026", look="R", slam=10, burst=True), 1: "CF042", 2: "L0040",
              4: dict(code="PM078", look="M", manga="impact", tone="ink", caption="アルトリア"), 6: "CF018",
              7: "7R184", 9: dict(code="A2030", look="Cd"), 11: "AS075", 13: "7R178",
              14: dict(code="OJ001", look="M", manga="impact", caption="沖田総司"),
              15: dict(code="OY002", look="R", det=["face", "head"]), 16: dict(code="OJ003", look="R", det=["face", "head"]),
              17: dict(code="OY003", look="Cd"), 18: dict(code="OA005", look="R"), 19: dict(code="OJ002", look="P", hold=12),
              20: dict(code="OY001", look="R", det=["person"], impact=3, slam=10, burst=True),     # 96.85 "And cast the first die!"
              21: dict(code="RN003", look="R")}),
    dict(seg="E08", b0=167, b1=184, cuts=[167, 171, 175, 179, 182], kind="lyrical", src=["MU"],
         pin={0: dict(code="MU002", look="M", manga="band", caption_lyr=45, tag="MIYAMOTO MUSASHI"),
              1: dict(code="MU003", look="R", det=["person"]),
              2: dict(code="MU001", look="M", manga="impact", tone="ink", caption="宮本武蔵"),
              3: dict(code="MU005", look="B", speed=0.4, poster=dict(big="MUSASHI", sub="Miyamoto Musashi  //  Saber", jp="宮本武蔵", num="03")),
              4: dict(code="MU004", look="Cd")}),
    dict(seg="E09", b0=184, b1=187, cuts=[184, 185, 186], kind="punch", src=["RS", "SK", "AT"], hot=True,
         pin={0: dict(code="RS037", look="P", hold=12), 1: dict(code="SK003", look="P", hold=12), 2: dict(code="AT004", look="P", hold=12)}),
    dict(seg="E10", b0=187, b1=219, cuts=[187, 191, 195, 199, 203, 205, 207, 209, 211] + list(range(212, 219)), kind="build",
         src=["RS", "RV", "CF", "SA"], hot=True,
         pin={**{0: "RS034", 1: dict(code="OA001", look="R", manual=[700, 200, 1900, 1100]), 2: "CF037", 3: dict(code="CF099", look="Cd"), 4: "RS029", 5: "SA052",
                  6: dict(code="CF039", look="P", hold=12), 7: "RV122", 8: "KK011"},
              **{9 + k: PB[212 + k] for k in range(7)}}),
    dict(seg="E11", b0=219, b1=283, cuts=[219, 220, 221, 222, 224] + rng(226, 243, 2) + rng(243, 267, 2) + rng(267, 279, 1) + [279, 280],
         kind="climax", src=["FS", "DL", "KK", "RV", "SA", "RN", "RA", "SK", "PM", "CF", "62", "2J", "DE", "YU", "QI", "A2", "AK", "AT", "AS"],
         hot=True,
         pin={**{k: PB[b] for k, b in enumerate([219, 220, 221, 222, 224])},
              **{5: "SA040", 7: "62034", 8: "FS053", 9: "AK053", 11: "RA012", 12: "DL038",
                     # 下面按拍号钉（"@拍号"），E11 中段插了一刀，槽位序号会漂
                     "@243": dict(code="AT051", look="P", impact=3, slam=10, burst=True),   # 142.106 "Cuts and bruises 超俗した…"
                     "@245": "A2039", "@247": "AS104",                                       # UR033 / 7R038 源里有闪（QC 144.2–145.4）
                     "@249": dict(code="RN001", look="P", impact=3, slam=10, burst=True),   # 145.612 "Fist emblazoned"
                     "@251": "AS082", "@253": "QI018", "@255": "2J034", "@259": "SK016",
                     "@265": dict(code="OA004", look="M", manga="impact"), "@270": dict(code="OA014", look="P", hold=12),
                     "@273": dict(code="OA003", look="R"), "@271": "PR028", "@263": "PR026", "@272": dict(code="PR039", look="Cd"), "@261": "DE026",     # HF3 宣传片独有：剑刃刺出 / 天马 / 黑樱哭喊
                     "@279": dict(code="2J036", look="P", hold=12),     # 163.10 贝德维尔举起金剑（归还圣剑）→ 接阿尔托莉雅终幕
                     "@280": dict(code="L0076", look="R", impact=3, slam=14, burst=True)}}),  # 163.68 "Judge his master" 终极：阿尔托莉雅
    dict(seg="E12", b0=283, b1="END", cuts=[283], kind="outro", src=["11"],
         pin={0: dict(code="11065", look="Cs")}),          # 马修回眸：不抠头，柔边大椭圆（用户："不用完全将遮罩缩小在头部范围"）
]
GLITCH = [(124.2, 127.0, 0.0, 0.22), (127.0, 127.57, 0.22, 0.35), (127.57, 128.128, 0.35, 0.9), (128.128, 128.27, 1.0, 0.0),
          (73.9, 74.699, 0.1, 0.6), (74.699, 74.84, 1.0, 0.0),            # E05 段尾 "Judge his master"
          (95.9, 96.85, 0.1, 0.6), (96.85, 96.99, 1.0, 0.0),              # E07 段尾 "And cast the first die!"
          (140.9, 142.106, 0.12, 0.6), (142.106, 142.25, 1.0, 0.0),      # "Cuts and bruises 超俗した…"
          (144.9, 145.612, 0.15, 0.6), (145.612, 145.75, 1.0, 0.0),      # "Fist emblazoned"
          (162.3, 163.68, 0.15, 0.9), (163.68, 163.83, 1.0, 0.0)]        # 终极 "Judge his master 破滅問う原罪"
PUNCH = [74.699, 96.85, 128.128, 142.106, 145.612, 163.68]
WALL = {55: "last", 64: True, 73: [1, 2, 3], 32: True, 44: "last", 31: "last", 46: "last"}
FADE = [(169.2, 171.0)]
HOT_LYR = set(range(29, 32)) | {13, 32} | set(range(42, 45)) | {46} | set(range(53, 74))


# ---------------------------------------------------------------- 选镜
L0PM = {}
for m in AUD.get("l0_to_pm", []):
    if isinstance(m, dict) and "l0_n" in m:
        L0PM[m["l0_n"]] = m


def upgrade(rec):
    """L0（FGO Memorial 1080）→ PM（英文版 4K 1440，同一画面平移 9.79 s）；有拖影备注的不换。"""
    if rec["yid"] != "L02aPFAhTCk" or rec["n"] not in L0PM:
        return rec
    m = L0PM[rec["n"]]
    if m.get("pm_note"):
        return rec
    r = dict(rec)
    r.update(yid="PmOY1wzlRjo", res=1440, **{"in": m["pm_in"], "out": m["pm_out"]})
    r["dur"] = r["out"] - r["in"]; r["code"] = rec["code"] + ">PM"
    return r


def score(r, seg, slot, used, last):
    if r["code"] in used or not r["usable"]:
        return -1e9
    s = {"A": 3.0, "B": 1.5, "C": 0.2}[r["grade"]]
    s += 1.6 * r["prio"] + (1.2 if r["res"] >= 1440 else 0.0) + (1.0 if r["fav"] else 0.0) - (6.0 if r["gil"] else 0.0)
    want = {"battle": ("战斗", "宝具", "特写"), "climax": ("宝具", "战斗", "特写"), "build": ("特写", "战斗", "宝具"),
            "lyrical": ("抒情", "特写"), "intro": ("特写", "宝具"), "punch": ("特写",), "outro": ("抒情",), "breath": ("特写",)}[seg["kind"]]
    s += 0.9 if r["kind"] in want else -0.5
    d = r["dur"] - 2 / 60.0
    s += 1.0 if d >= slot else (0.4 if d / slot >= 0.6 else -2.0 * (1 - d / slot))
    s -= 1.5 if r["dark"] >= 0.5 or r["yavg"] < 30 else 0.0
    s -= 2.0 if r["split"] else 0.0
    s -= 2.5 if any(("%s%03d" % (r["code"][:2], r["n"] + k)) in used for k in (-1, 1)) else 0.0
    if last is not None and last["yid"] == r["yid"]:
        s += 0.4 if r["n"] > last["n"] else -0.8
    return s


def pick(seg, slot, used, last, run):
    cands = [r for r in POOL.values() if r["code"][:2] in seg["src"]]
    if run >= 3 and last is not None:
        cands = [c for c in cands if c["yid"] != last["yid"]] or cands
    return max(cands, key=lambda c: score(c, seg, slot, used, last))


def look_for(r, seg, slot, i, prev_looks, b):
    """用户："全局都要风格化（特别是第三种 causasui）…两种都用"：人物特写 / 抒情 → causasui（Cd 暗态 / Cb 档案），
    战斗 / 宝具 → Raf0x00 HUD（R），高潮重拍 → hectoroz 版画（P）。目标比例约 R 55% / Cd 25% / Cb 8% / P 12%。"""
    k, who = r["kind"], r["who"]
    has_char = who not in ("none", "", None) and "unclear" not in who
    hot = seg.get("hot") and impact(b) >= 0.55 and seg["kind"] in ("climax", "build", "battle")
    if seg["kind"] in ("lyrical", "outro"):
        return "Cb" if (has_char and slot >= 1.6 and "Cb" not in prev_looks[-1:]) else "Cd"
    if hot and k in ("宝具", "战斗", "特写") and prev_looks[-1:] != ["P"]:
        return "P"
    since_cb = next((n for n, x in enumerate(reversed(prev_looks)) if x == "Cb"), 99)
    if has_char and slot >= 1.1 and since_cb >= 8 and k in ("特写", "抒情"):          # 档案页要能裁到脸：不给远景 / 动作镜头
        return "Cb"
    if has_char and k in ("特写", "抒情") and prev_looks[-1:] != ["Cd"]:
        return "Cd"
    if has_char and k == "战斗" and slot >= 0.9 and prev_looks[-2:] == ["R", "R"]:
        return "Cd"
    return "R"


def look_v5(r, seg, slot, i, prev_looks, b):
    """v5（用户第 2 条 "HUD 覆盖时间太长了，可以引入一些先前的漫画分镜风格化处理方式"）：
    ≥ 0.9 s 的 R / Cd 槽里每隔两镜换一张漫画页 M；高潮 / 蓄力段隔三镜；1 拍一切的密集段不用（分镜要时间读）。"""
    look = look_for(r, seg, slot, i, prev_looks, b)
    if look not in ("R", "Cd") or slot < 0.9 or seg["kind"] in ("punch", "breath", "outro"):
        return look
    since = next((n for n, x in enumerate(reversed(prev_looks)) if x == "M"), 99)
    gap = 8 if seg["kind"] in ("climax", "build") else 6       # v7："黑白漫画少用"：漫画页频率减半
    if since >= gap and r["dark"] < 0.5 and r["yavg"] >= 30:      # 太暗的源转灰阶网点会糊成一片
        return "M"
    return look


import hud_v4 as H4
_NA = {"A": -99, "nA": 0, "G": set(), "i": 0}
NO_COMIC = {"okita", "okita_alter", "musashi", "jeanne"}     # 用户："冲田总司，宫本武藏以及贞德的画面不要用美漫处理"


def char_key(who):
    inf = H4.info_for(who, "", [])
    return inf.get("key") if inf else None


def look_v6(r, seg, slot, i, prev_looks, b):
    """v6：在 v5 上加 G 灵基卡（关键英灵的特写，每人一次，<= 6）和 A 美漫（战斗 / 宝具，间隔 >= 7 镜，<= 9）。"""
    _NA["i"] += 1
    look = look_v5(r, seg, slot, i, prev_looks, b)
    key = char_key(r["who"])
    calm = seg["kind"] not in ("punch", "breath", "outro") and not (seg.get("hot") and impact(b) >= 0.55)
    if (look in ("R", "Cd", "M", "Cb") and calm and slot >= 1.1 and key and key not in _NA["G"] and len(_NA["G"]) < 6
            and r["kind"] in ("特写", "抒情") and r["dark"] < 0.5):
        _NA["G"].add(key)
        return "G"
    if (look in ("R", "M") and slot >= 0.9 and r["kind"] in ("战斗", "宝具") and _NA["i"] - _NA["A"] >= 5 and _NA["nA"] < 14
            and key not in NO_COMIC
            and r["dark"] < 0.5 and r["yavg"] >= 45):
        _NA["A"] = _NA["i"]; _NA["nA"] += 1
        return "A"
    return look


HUD_T = ("lock", "dossier", "type", "hexwall", "np", "scope")


def assign_hud(shots):
    """R / P 镜头的 HUD 模板：按信息量和时长筛可用模板，取用得最少的、且不和上一个 HUD 镜头重复（钉子里写了 hud 的不动）。"""
    use = {k: 0 for k in HUD_T}
    prev = None
    rng = np.random.default_rng(11)
    for sh in shots:
        if sh["look"] not in ("R", "P"):
            continue
        if sh.get("hud"):
            use[sh["hud"]] = use.get(sh["hud"], 0) + 1; prev = sh["hud"]; continue
        inf = H4.info_for(sh["who"], sh.get("title", ""), sh.get("tags", []))
        slot = sh["b"] - sh["a"]
        ok = ["lock", "scope", "hexwall", "blueprint"] + (["type"] if inf else []) + (["codex"] if inf and inf.get("jp") and slot >= 1.0 else [])   # v10：+ blueprint / codex
        if inf and inf.get("p") and slot >= 1.1:
            ok.append("dossier")
        if inf and (inf.get("np_jp") or inf.get("np_en")) and slot >= 0.9:
            ok.append("np")
        if slot < 0.75:
            ok = [k for k in ok if k in ("lock", "scope", "type")]
        ok = [k for k in ok if k != prev] or ok
        k = min(ok, key=lambda x: (use[x], rng.random()))
        sh["hud"] = k; use[k] += 1; prev = k
    print("  HUD 模板:", use)


SLIT_NAME = {"artoria": "ARTORIA", "artoria_alter": "SABER ALTER", "jeanne": "JEANNE", "okita": "OKITA", "okita_alter": "OKITA ALTER", "musashi": "MUSASHI",
             "medusa": "MEDUSA", "karna": "KARNA", "mordred": "MORDRED", "mash": "MASH", "richard": "LIONHEART"}


def add_slits(shots, trans):
    """画幅收窄 + 强调字转场（用户："IMPERSONATOR 这个缩窄画幅然后强调文字的转场我很喜欢，可以稍微多做一些"）：
    切到一个新出场的关键英灵时，缝里展开名字（下一行职阶 + 日文名）；间隔 >= 9 s，最多 10 个；替换原来那个转场。"""
    last, n, seen = -99.0, 0, set()
    at = {round(t["t"], 3): t for t in trans}
    for k in range(1, len(shots)):
        A_, B_ = shots[k - 1], shots[k]
        fa, fb = (A_["b"] - A_["a"]) * FPS, (B_["b"] - B_["a"]) * FPS
        if B_.get("impact") or A_.get("bars") or (B_.get("speed") == 0.0 and B_.get("bars")) or fa < 24 or fb < 40:
            continue
        kb = char_key(B_["who"]); ka = char_key(A_["who"])
        if not kb or kb == ka or kb in seen or B_["a"] - last < 9.0 or n >= 10 or B_["look"] in ("B", "M"):
            continue
        inf = H4.info_for(B_["who"], "", [])
        at[round(B_["a"], 3)] = dict(t=B_["a"], kind="slit", pre=12, post=18, text=SLIT_NAME[kb],
                                     sub="CLASS  %s   /   %s" % (inf["cls"], inf.get("jp", "")))
        last = B_["a"]; n += 1; seen.add(kb)
    print("  slit:", n)
    return sorted(at.values(), key=lambda t: t["t"])


import palette as PALT


def duo_of(r):
    """两人同框打斗 → 两人各自的签名色 [A, B]（色相太近就把 B 换成 B 的副色）；否则 None。"""
    names = [n.strip() for n in re.split(r"[、,，；;/+]", r["who"] or "") if n.strip() and "unclear" not in n]
    fight = r["kind"] in ("战斗", "宝具") or re.search("交锋|对决|互殴|相抵|对打|交战|激斗", r["what"] or "")
    if len(names) < 2 or not fight:
        return None
    ca, cb = PALT.colors_of(names[0]), PALT.colors_of(names[1])
    if not ca or not cb:
        return None
    A, B = list(ca[0]), list(cb[0])
    ha, hb = PALT._hsv(A)[0] * 360, PALT._hsv(B)[0] * 360
    if min(abs(ha - hb), 360 - abs(ha - hb)) < 40:
        B = list(cb[1])
    return [A, B]


def manga_of(r, slot, seg, n_m):
    """漫画版式：战斗 / 宝具 → impact（≥1 s，主格描线网点 + 速度线）或 slant；特写 / 抒情 → story（≥1.6 s）/ band。"""
    k = r["kind"]
    if k in ("战斗", "宝具"):
        return ("impact" if slot >= 1.0 and n_m % 2 == 0 else "slant"), "ink"
    if slot >= 1.6 and k in ("特写", "抒情", "群像"):
        return "story", "gray"
    return ("band" if n_m % 2 == 0 else "slant"), "gray"


def make_shot(r, a, b, pin, seg, look):
    t0, t1 = bt(a), bt(b)
    slot = t1 - t0
    sh = dict(id="%s_%02d" % (seg["seg"], 0), code=r["code"], yid=r["yid"], look=look, a=round(t0, 4), b=round(t1, 4),
              what=r["what"], who=r["who"], kind=r["kind"], grade=r["grade"], res=r["res"], group=r["group"])
    s_in, s_out = r["in"] + 1 / 60.0, r["out"] - 1 / 60.0
    if s_out - s_in < 0.15:
        s_in, s_out = r["in"], r["out"]
    if "src_a" in pin:
        sh.update(src_a=pin["src_a"], speed=pin.get("speed", 1.0), lo=pin.get("lo", s_in), hi=pin.get("hi", s_out))
    else:
        need = slot + 0.4                                   # 前后转场余量
        avail = s_out - s_in
        min_sp = 0.5 if r["res"] >= 1440 else 0.6
        sp = 1.0 if avail >= need else max(min_sp, avail / need)
        if pin.get("speed") is not None:
            sp = pin["speed"]
        lead = 0.2 * sp
        extra = max(0.0, avail - need * max(sp, 1e-6))
        src_a = s_in + lead + extra * 0.35
        sh.update(src_a=round(src_a, 4), speed=round(sp, 4), lo=round(s_in, 4), hi=round(s_out, 4))
    if r.get("crop"):
        sh["crop"] = r["crop"]; sh["crop_anchor"] = r.get("crop_anchor", "top")
    title, tags, labs = char_of(r["who"], r["what"])
    if look in ("R", "P"):
        sh["det"] = pin.get("det") or (["face", "head", "person"] if r["kind"] == "特写" else ["person", "head", "face"])
        sh["manual"] = pin.get("manual") or [W * 0.33, H * 0.18, W * 0.67, H * 0.82]
        sh["title"] = pin.get("title") or title
        sh["tags"] = pin.get("tags") or tags
        if look == "P":
            sh.setdefault("hold", pin.get("hold", 12))
    if look == "Cd":
        sh["labels"] = labs
    if look == "Cb":
        names = [n.strip() for n in re.split(r"[、,，/+与]", r["who"]) if n.strip() and "unclear" not in n]
        sh["titles"] = pin.get("titles") or ["%s // %s" % (char_of(n)[0], "RECORD %02d" % (j + 1)) for j, n in enumerate(names[:2])] or ["ARCHIVE // RECORD"]
        bb = [x for x in B if x > t0 + 0.25][:3]
        sh["box_beats"] = [round(bb[0], 3), round(bb[1], 3)] if len(bb) >= 2 else [t0 + 0.3, t0 + 0.9]
    if look == "M":
        _NM["n"] += 1
        kind_, tone_ = manga_of(r, slot, seg, _NM["n"])
        sh["manga"] = pin.get("manga") or kind_
        sh["tone"] = pin.get("tone") or "mono"               # v8："漫画的全彩可以改为 monocolor，多用单色，黑白占比少一点"
        duo = duo_of(r)
        if duo and not pin.get("tone"):                     # v9："双人对打的漫画风格化处理可以改成双色"
            sh["tone"] = "duo"; sh["duo"] = duo
        sh["num"] = _NM["n"] + 6
        sh["tag"] = pin.get("tag") or ("%s  //  %s" % (title, " ".join(tags[0].split()[1:])) if title != "UNIDENTIFIED" else "SUBJECT  //  UNKNOWN")
        if "caption" in pin or "caption_lyr" in pin:
            for k_ in ("caption", "caption_lyr"):
                if k_ in pin:
                    sh[k_] = pin[k_]
        elif sh["manga"] in ("story", "impact") and jp_of(r["who"]):
            sh["caption"] = jp_of(r["who"])
        else:                                               # 日文歌词行恰好从这镜开头唱起 → 进竖排说明框（文字由引擎从用户包取）
            for li, row in enumerate(LYR):
                st = row["start"] / 1000.0
                if t0 - 0.15 <= st <= t0 + 0.4 and li not in HOT_LYR and not re.search(r"[A-Za-z]", row["text"]) and len(row["text"]) <= 16:
                    sh["caption_lyr"] = li
                    break
    if look == "B":
        sh["poster"] = pin["poster"]
    if look == "A":                                         # 美漫：说明框只写核对过的真名 / 职阶；冲击镜头加拟声字
        inf = H4.info_for(r["who"], title, tags)
        if inf and inf.get("key"):
            sh["caption"] = inf["en"]; sh["cap_tag"] = inf["cls"]
        w_ = r["what"]
        sh["sfx"] = pin.get("sfx") or ("BOOM!" if re.search("爆|炸", w_) else "CLANG!" if re.search("交锋|相抵|格挡|火花", w_) else
                                       "SLASH!" if re.search("斩|挥|劈|刀|剑", w_) else "FWOOSH!" if re.search("光|宝具|冲", w_) else "KRAK!")
    if pin.get("hud"):
        sh["hud"] = pin["hud"]
    for k_ in ("hold", "boil", "hud_rate", "growth", "tension", "push", "bars", "impact", "slam", "burst", "tear", "cropbox", "keep_x", "vs"):
        if k_ in pin:
            sh[k_] = pin[k_]
    return sh


_NM = {"n": 0}


# ---------------------------------------------------------------- 转场
def assign_trans(shots, segs_of):
    trans = []
    cyc = 0
    hx = {"n": 0}
    for k in range(1, len(shots)):
        A, Bs = shots[k - 1], shots[k]
        fa, fb = (A["b"] - A["a"]) * FPS, (Bs["b"] - Bs["a"]) * FPS
        sk = segs_of[k]
        t = Bs["a"]
        bi = min(range(len(B)), key=lambda i: abs(B[i] - t))
        imp = impact(bi)
        if Bs.get("impact") or (Bs.get("speed") == 0.0 and Bs.get("bars")):
            continue                                        # 爆点 / 换气：硬切（冲击帧自己就是转场）
        if A.get("bars"):
            continue
        if Bs["look"] == "Cb" and min(fa, fb) >= 30:
            trans.append(dict(t=t, kind="panel_slap", pre=12, post=12)); continue
        m = min(fa, fb)
        if Bs["look"] in ("M", "B") or A["look"] in ("M", "B"):     # 漫画页 / 海报自己会拍格入场：蜂巢翻页 / glitch / glitch 切轮换
            kind = ["hex", "whip", "glitch", "grid", "hex", "gcut"][hx["n"] % 6] if m >= 36 else "gcut"
            hx["n"] += 1
            pre, post = {"hex": (9, 9), "whip": (6, 6), "grid": (7, 7), "glitch": (5, 6), "gcut": (2, 3)}[kind]
            trans.append(dict(t=t, kind=kind, pre=pre, post=post)); continue
        if sk in ("lyrical", "outro", "intro"):
            opts = ["lock_zoom", "hex", "grid", "whip"] if m >= 40 else ["whip", "gcut"]
            kind = opts[cyc % len(opts)]; cyc += 1
            if kind == "lock_zoom" and A["look"] not in ("R", "P", "Cd"):
                kind = "grid"
            pre, post = {"lock_zoom": (12, 6), "hex": (9, 9), "grid": (8, 8), "whip": (7, 7), "gcut": (2, 3)}[kind]
            trans.append(dict(t=t, kind=kind, pre=pre, post=post)); continue
        if m < 40:                                          # 1 拍一切：全部 glitch 切（v9："glitch 效果可以多加一点"）
            trans.append(dict(t=t, kind="gcut", pre=2, post=3))
            cyc += 1; continue
        if imp >= 0.6 and Bs["look"] in ("R", "P"):
            trans.append(dict(t=t, kind="glitch_slam", pre=4, post=8)); continue
        opts = ["glitch", "hex", "whip", "lock_zoom", "grid", "gcut", "cut"]
        kind = opts[cyc % len(opts)]; cyc += 1
        if kind == "lock_zoom" and (A["look"] not in ("R", "P") or m < 50):
            kind = "whip"
        if kind == "cut":
            continue
        pre, post = {"glitch": (5, 6), "hex": (9, 9), "whip": (6, 6), "lock_zoom": (12, 6), "grid": (7, 7), "gcut": (2, 3)}[kind]
        trans.append(dict(t=t, kind=kind, pre=pre, post=post, dir=1 if cyc % 2 else -1))
    return trans


# ---------------------------------------------------------------- 歌词
def lyric_map(shots):
    out = {}
    for i, r in enumerate(LYR):
        t = r["start"] / 1000.0
        sh = next((s for s in shots if s["a"] <= t + 0.05 < s["b"]), shots[-1])
        txt = r["text"]
        cjk = any(ord(c) >= 0x3000 for c in txt)
        en = re.search(r"[A-Za-z]", txt)
        hot = i in HOT_LYR
        if sh.get("caption_lyr") == i:                      # 已进漫画竖排说明框
            out[i] = dict(style="caption")
        elif hot or (en and not cjk):
            out[i] = dict(style="burst", hot=hot)
            if i in WALL:
                out[i]["wall"] = WALL[i]
        elif sh["look"] in ("Cd", "G", "Cs"):
            out[i] = dict(style="serif_dark")
        elif sh["look"] in ("Cb", "M", "B", "A"):           # 纸面：墨色明朝
            out[i] = dict(style="serif_ink")
        else:
            out[i] = dict(style="hud", pos=[0.06, 0.80])
    return out


# ---------------------------------------------------------------- 闪烁预检（源亮度，按成片时间映射）
def luma_track(sh):
    path = __import__("cards").src_path(sh["yid"])
    a, b = sh["lo"], sh["hi"]
    p = subprocess.run([FF, "-v", "error", "-ss", "%.3f" % a, "-t", "%.3f" % (b - a + 0.05), "-i", path, "-vf", "scale=48:27",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    fr_ = np.frombuffer(p.stdout, np.uint8)
    n = len(fr_) // (48 * 27)
    if n == 0:
        return None, None
    f = fr_[:n * 48 * 27].reshape(n, 27, 48).astype(np.float32)
    return f.mean((1, 2)), np.percentile(f.reshape(n, -1), 99.5, axis=1)


def flash_check(shots):
    warn = []
    for sh in shots:
        if sh.get("speed", 1) == 0:
            continue
        m, hi = luma_track(sh)
        if m is None:
            warn.append((sh["code"], "decode fail")); continue
        fps_src = len(m) / max(1e-3, (sh["hi"] - sh["lo"] + 0.05))

        def bad(src_a):
            ts = [src_a + (F / FPS - sh["a"]) * sh["speed"] for F in range(int(sh["a"] * FPS) - 6, int(sh["b"] * FPS) + 6)]
            idx = [int(np.clip((t - sh["lo"]) * fps_src, 0, len(m) - 1)) for t in ts]
            y = m[idx]; h = hi[idx]
            dy = np.abs(np.diff(y))
            jumps = int((dy >= 30).sum())
            win = max([int((dy[k:k + 60] >= 30).sum()) for k in range(0, max(1, len(dy) - 59), 10)] or [0])
            jumps += 10 if win > 2 else 0
            black = int(((y < 4) & (h < 24)).sum()) + int(((y < 16) & (h < 70)).sum())
            out = int(sum(1 for t in ts if t > sh["hi"] + 0.02 or t < sh["lo"] - 0.02))
            return jumps * 3 + black * 5 + out * 0.05
        b0 = bad(sh["src_a"])
        if b0 >= 3:
            span = sh["hi"] - sh["lo"] - (sh["b"] - sh["a"]) * sh["speed"]
            best = (b0, sh["src_a"])
            for d in np.linspace(sh["lo"] + 0.1, sh["lo"] + max(0.1, span), 12):
                v = bad(d)
                if v < best[0]:
                    best = (v, d)
            if best[1] != sh["src_a"]:
                warn.append((sh["code"], "slide %.2f→%.2f (bad %.0f→%.0f)" % (sh["src_a"], best[1], b0, best[0])))
                sh["src_a"] = round(float(best[1]), 4)
            if best[0] >= 3:
                warn.append((sh["code"], "仍有闪/黑帧风险 %.0f" % best[0]))
            if best[0] >= 15:
                sh["_risky"] = best[0]
    return warn


# ---------------------------------------------------------------- 主流程
_SIG = {}


def sig(sh):
    key = (sh["yid"], round(sh["src_a"], 2))
    if key in _SIG:
        return _SIG[key]
    from cards import src_path
    out = []
    for q in (0.25, 0.5, 0.75):
        t = min(sh["hi"], sh["src_a"] + (sh["b"] - sh["a"]) * sh.get("speed", 1) * q)
        p = subprocess.run([FF, "-v", "error", "-ss", "%.3f" % t, "-i", src_path(sh["yid"]), "-frames:v", "1", "-vf", "scale=64:36",
                            "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
        v = np.frombuffer(p.stdout, np.uint8)[:64 * 36].astype(np.float32)
        if v.size == 64 * 36:
            v = (v - v.mean()) / (v.std() + 1e-3)
            out.append(v)
    _SIG[key] = out
    return out


def dupes(shots, pinned):
    bad = []
    S = [sig(s) for s in shots]
    for i in range(len(shots)):
        for j in range(i + 1, len(shots)):
            if shots[i]["code"].split(">")[0] == shots[j]["code"].split(">")[0]:
                continue
            c = max((float((a * b).mean()) for a in S[i] for b in S[j]), default=0.0)
            if c > 0.92:
                for k in (j, i):
                    cd = shots[k]["code"].split(">")[0]
                    if cd not in pinned:
                        bad.append(cd); break
                else:
                    print("  ! 两个钉子画面重复：%s / %s（%.2f）" % (shots[i]["code"], shots[j]["code"], c))
    return bad


# 女性台词（用户第 8 条）：只放在人声空档（0–8.78 / 30.98–38.24 / 100.89–108.24 / 165.73–）；
# t = 成片时间，(yid, s0, s1) = 源里的句子区间（dialog_harvest 切出、人声分离后的干声）；文字不存这里
# v6（用户："台词…对不上人物，例如片尾马修的台词是贞德的"）：每句只放在说话人在画面上的镜头里；
# 30.98–38.24 / 100.89–108.24 两段其实有合唱，台词混音时对音乐中频做 M/S 挖槽（不整体压低音乐）
# v7：28–38.2 s 是 JUST LIKE ALL THE REST 的慢唱、100.5–102.7 也在唱（干声音高扫描），台词不再放进去
DIALOG = [dict(t=2.58, yid="Cf9uMIffDTQ", s0=85.87, s1=88.15, who="贞德", note="E01 同步口型"),
          dict(t=103.20, yid="MUkengo4K60", s0=1.68, s1=2.72, who="宫本武藏", note="MU001 竹林（同源画面）；102.8–107.5 干声无音高"),
          dict(t=105.40, yid="MUkengo4K60", s0=6.96, s1=7.80, who="宫本武藏", note="MU005 武藏面部海报"),
          dict(t=166.00, yid="Cf9uMIffDTQ", s0=114.80, s1=116.12, who="玛修", note="片尾马修画外音"),
          dict(t=167.75, yid="Cf9uMIffDTQ", s0=116.70, s1=117.80, who="玛修", note=""),
          dict(t=169.00, yid="Cf9uMIffDTQ", s0=118.18, s1=119.05, who="玛修", note="")]
NO_TRANS = {50.09}   # 成片 QC：2J002 源内两次闪 + whip 进漫画纸面两次跳变 = 4 次/秒 → 这里硬切
def beat_glitch():
    """强拍短 glitch 脉冲（v9）：能量 >= 0.48 的拍、不在已有包络 ±0.6 s 内 → 0.55 起、8 帧落回 0；两次之间 >= 1.2 s。"""
    out, last = [], -9.0
    for b in range(len(B)):
        t = B[b]
        if impact(b) < 0.48 or t - last < 1.5 or t > SONG_END - 4:
            continue
        if any(t0 - 0.6 <= t <= t1 + 0.6 for t0, t1, _, _ in GLITCH):
            continue
        out.append((round(t, 3), round(t + 8 / 60.0, 3), 0.55, 0.0)); last = t
    print("  强拍 glitch:", len(out))
    return out


BAN = {"SB038", "KK023"}   # KK023：末 3 帧有葡语字幕          # 成片 QC：源里频闪（46.6–47.1 s 11 次/秒）


def build(banned=()):
    _NM["n"] = 0
    _NA.update(A=-99, nA=0, G=set(), i=0)
    used = set(banned) | BAN
    for sg in SEGS:                                         # 钉子先占用
        for p in sg["pin"].values():
            used.add((p if isinstance(p, str) else p["code"]))
    shots, segs_of, warn = [], [], []
    for sg in SEGS:
        cuts = list(sg["cuts"]) + [sg["b1"]]
        sg["pin"] = {(cuts.index(int(k[1:])) if isinstance(k, str) else k): v for k, v in sg["pin"].items()}
        last, run, prev_looks = None, 0, [s["look"] for s in shots[-2:]]
        for si, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            slot = bt(b) - bt(a)
            pin = sg["pin"].get(si)
            if pin is not None:
                pin = {"code": pin} if isinstance(pin, str) else dict(pin)
                code = pin["code"]
                if code not in POOL:
                    warn.append("%s 槽%d 钉子 %s 不在可用池" % (sg["seg"], si, code))
                    r = pick(sg, slot, used, last, run)
                else:
                    r = POOL[code]
            else:
                pin = {}
                r = pick(sg, slot, used, last, run)
            used.add(r["code"])
            if "src_a" not in pin:
                r = upgrade(r)
            look = pin.get("look") or look_v6(r, sg, slot, si, prev_looks, a)
            sh = make_shot(r, a, b, pin, sg, look)
            sh["id"] = "%s_%02d" % (sg["seg"], si)
            sh["seg"] = sg["seg"]
            sh["hot"] = bool(sg.get("hot"))
            shots.append(sh); segs_of.append(sg["kind"]); prev_looks.append(look)
            run = run + 1 if (last is not None and last["yid"] == r["yid"]) else 1
            last = r
    assign_hud(shots)
    trans = assign_trans(shots, segs_of)
    trans = [t for t in trans if round(t["t"], 2) not in NO_TRANS]
    trans = add_slits(shots, trans)
    fw = flash_check(shots)
    pinned = {(p if isinstance(p, str) else p["code"]) for sg in SEGS for p in sg["pin"].values()}
    risky = [s["code"].split(">")[0] for s in shots if s.get("_risky") and s["code"].split(">")[0] not in pinned]
    dd = dupes(shots, pinned)
    if dd:
        print("  画面重复、换镜重排:", dd)
    risky += dd
    if risky and len(banned) < 40:
        print("  闪烁风险大、换镜重排:", risky)
        return build(tuple(banned) + tuple(risky))
    for s in shots:
        s.pop("_risky", None)
    lyr = lyric_map(shots)
    cfg = dict(T0=0.0, T1=SONG_END, out="PROVANT_AMV_v7_video.mp4", shots=shots, trans=trans, lyrics=lyr,
               glitch=GLITCH + beat_glitch(), punch=PUNCH, fade=FADE, dialog=DIALOG, hud_v=5)
    json.dump(cfg, open(os.path.join(ROOT, "data", "edl_v9.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # ---- 统计 + 人看的表
    n = len(shots)
    tk = Counter(t["kind"] for t in trans)
    hard = n - 1 - len(trans)
    lk = Counter(s["look"] for s in shots)
    gp = Counter(s["group"] for s in shots)
    res = Counter(s["res"] for s in shots)
    Ls = ["# EDL v9（自动生成：scripts/edl_v9.py）", "",
          "镜头 %d；风格 %s；转场 %s，硬切 %d（%.0f%%）；系列 %s；分辨率 %s" % (
              n, dict(lk), dict(tk), hard, 100.0 * hard / max(1, n - 1), dict(gp), dict(res)), ""]
    for s in shots:
        Ls.append("- `%s` %7.3f–%7.3f %-4s %-10s %s/%s %s — %s（%s）" % (
            s["id"], s["a"], s["b"], s["look"], s["code"], s["group"], s["res"], s["kind"], s["what"][:40], s["who"][:16]))
    Ls += ["", "## 预检提示"] + ["- " + str(w) for w in warn + fw]
    open(os.path.join(ROOT, "EDL_v9.md"), "w", encoding="utf-8").write("\n".join(Ls))
    print(Ls[2])
    for w in warn + fw:
        print("  !", w)
    return cfg


def sheets(cfg):
    from cards import src_path
    out = os.path.join(ROOT, "ae", "qa", "v9"); os.makedirs(out, exist_ok=True)
    from PIL import Image, ImageDraw, ImageFont
    fnt = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 15)
    by = {}
    for s in cfg["shots"]:
        by.setdefault(s["seg"][:3], []).append(s)
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
            d.text((x + 4, y), "%s %s %s %.2f %s" % (s["id"][-2:], s["look"], s["code"], s["a"], tr_at.get(round(s["a"], 3), "|")),
                   font=fnt, fill=(255, 220, 90))
        sheet.save(os.path.join(out, "%s.jpg" % seg), quality=85)
    print("sheets ->", out)


if __name__ == "__main__":
    cfg = build()
    if "--no-sheets" not in sys.argv:
        sheets(cfg)

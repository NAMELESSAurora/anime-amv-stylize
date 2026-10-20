# -*- coding: utf-8 -*-
"""
PROVANT v10 成片 EDL（v12）→ data/edl_v12.json
  画面：.venv_gpu python scripts/v2_preview.py V12 analyze|render|encode → out/v10
  混音：.venv_gpu python scripts/v4_audio.py v12 --mux → out/PROVANT_AMV_v10.mp4（原声 / 环境底 / 爆点全部读去配乐版，scripts/sep_bandit.py）
在 v11（v9 成片）上按用户 10-20 看过 v9 的意见改：
  1. "美漫的处理风格缺少张力，一律替换成别的吧" → A 全部换掉：轨迹 HUD（track）/ 取景器 HUD（viewfinder）/ 双人标注 HUD（link）/
     漫画冲击页 / 推镜冲击 / 宝具横幅
  2. 冲田语音重复（0:43–0:47）→ v4_audio 环境底按源时间去重（同一段源声音全片只出现一次，台词用掉的区间让开）
  3. "将素材里的音乐…剥离，仅保留人物音和环境音" → v4_audio 全部改读去配乐版
  4. "贝德维尔的片段可以往后放一些（前面羽蛇神刚打完又暂停）" → break 1 回歌后先接武藏（78.45–88.25），
     贝德维尔挪到 88.25–107.50，第二次停歌挪到 107.50（歌里副歌 2 前唯一的自然低谷 107.3–108.2；两次停歌之间从 19 s 拉到 29 s）；
     贝德维尔的器乐段（100.89–107.50，无人声）放银臂 + "スイッチオン…アガートラム" 原声同步；
     break 2 改成回忆对话 → 胜利誓约剑 → 对波 → 狮子王最后一句 → 贝德维尔消散（源 38.00–74.60，一刀不剪）
     "0:02:39 的 HUD 没有对准人物"：2J049 源里有三个镜头（悬崖远景 / 两人对峙 / 狮子王特写），拆开；悬崖远景用手标框的双人标注（link）
  5. "HUD 种类也可以多设计和加入一些" → hud_v5 新增 track / viewfinder；看过 v10 后 versus（"出戏、幼稚"）删掉，换 link / blueprint / codex
  6. "给羽蛇神的片段和小贝的片段加上台词演出，台词做精致 motion 处理" → v6_title 新增 line 样式（说话人牌 + 逐字按说话时刻出现 + 进度线 +
     喊招砸字），台词文字来自去配乐干声的两个 ASR 模型交叉核对，说话人按画面口型 / 声音高低核对过，核对不了的不上字幕
"""
import json, os, re, subprocess, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import edl_v10 as E10
from edl_v10 import V9, PALT, INFO, POOL, bt, item, place, shot_of, fixed_shot, slit, OLD, OLD_TR
from edl_v9 import ROOT, FF, FPS, SONG_END, B
import edl_v11 as E11
from edl_v11 import RICH, HIPPO, LINE, BREAK_SONG, OKITA_LINES, MASH_TAIL, H_, R_, native_shot, run_codes, tape_for, PP

# 序章 → 标题（用户看过 v10 第二版："去掉 0:00:09 开场时的原音吼叫（洞窟画面）""或者可以换一下开场的片段，连着理查和绫香的画面，然后放 TITLE"）：
#   不再用洞口山丘（EP015）做标题底；竞技场废墟远景（EP126：左绫香、右 Richard 红披风）原速放到台词说完（394.80），
#   同一镜头接着慢放（394.80–395.45 摊满标题卡；源 395.5 起黑场 + 字幕表）当标题底，画面渐暗、标题从里面浮出（速度斜坡，不再 glitch 切）。
#   序章短 0.65 s、标题卡长 0.65 s → 歌的起点和后面全片不动
PROLOGUE = [x for x in E11.PROLOGUE if x[0] != "EP126"] + [("EP126", 393.54, 394.80, 1.0)]
TITLE_SRC = ("EP126", 394.80, 395.45)
TITLE_PRE = E11.TITLE_PRE + (395.45 - 394.80)

V9.SLIT_NAME.update(hippolyta="HIPPOLYTA")
MUSASHI_END = 88.25                                          # 第 38 行歌词起唱：武藏 → 贝德维尔
BREAK2_SONG = 107.50                                         # 副歌 2（108.24）前的自然低谷（107.3 起掉到 -21 dB）；歌在这里自己落下去，原声接上
BD_GAP = 100.89                                              # 第 45 行唱完 → 器乐段（无人声）：贝德维尔 "スイッチオン…アガートラム" 原声同步
# ---------------------------------------------------------------- break 1：魁札尔（v11 的 LOOK1，美漫全部替换）
LOOK1 = dict(E11.LOOK1)
LOOK1.update({"DE002": dict(look="R", hud="track", who="魁札尔·科亚特尔"),           # 火球俯冲：真实轨迹 + 速度 / 航向
              "DE009": dict(look="R", nohud=True, slam=10, burst=True),               # 摔角靴：推镜冲击
              "DE014": dict(look="R", hud="viewfinder", who="魁札尔·科亚特尔"),      # 火球拖尾划过灰天：取景器
              "DE026": dict(look="M", manga="impact", tone="mono", tag="QUETZALCOATL  //  RIDER"),   # 猛击粉色屏障：漫画冲击页（单色）
              "DE036": dict(look="R", nohud=True, slam=8)})                          # 粉色光束：推镜
# 用户："break 时…画面动态调整为 21:9，并且…点状纹理，改为电影质感滤镜"；随后澄清："是减少片段的点状画面纹理，将纹理换为电影质感滤镜
#   而不是去掉风格化（如漫画等）处理" → 漫画页 / riso / 取景器全部保留；break 镜头 cine=True：R 底不加网点、漫画 / riso 网点换连续调、取景器不画自己的遮幅
CINE_IN, CINE_OUT = 0.75, 0.6                                # 21:9 遮幅收拢 / 打开的时长（秒）
# ---------------------------------------------------------------- break 2：贝德维尔回忆 → 胜利誓约剑 → 对波 → 狮子王 → 消散
# 2J049 源里是三个镜头（帧差：63.625 硬切，67.4 起叠化到狮子王特写），拆成三条
SPLIT = {"2J049": [63.625, 67.40]}
LOOK2 = {"2J034": dict(look="R", nohud=True), "2J035": dict(look="R", nohud=True),
         "2J036": dict(look="R", nohud=True, slam=6),                                  # 金光长剑竖起（手化为剑）
         "2J037": dict(look="R", nohud=True),
         "2J038": dict(look="R", nohud=True), "2J039": dict(look="R", nohud=True),
         "2J040": dict(look="R", nohud=True, slam=8),                                  # v11 的美漫 → 推镜
         "2J041": dict(look="R", nohud=True), "2J042": dict(look="R", nohud=True), "2J043": dict(look="R", nohud=True),
         "2J044": dict(look="R", nohud=True), "2J045": dict(look="R", nohud=True), "2J046": dict(look="R", nohud=True),
         "2J047": dict(look="R", nohud=True), "2J048": dict(look="R", nohud=True),
         # 悬崖上两个小人影（左：戴冠、毛领披风 = 狮子王，和 67.4 起的特写同一身披风；右：贝德维尔）——手标框，不让检测锁到光带上
         "2J049#0": dict(look="R", hud="link", who="贝德维尔", det=["manual"], manual=[1060, 760, 1500, 1440], damp=True,
                         vs=dict(keys=["lion_king", "bedivere"], pts=[(0.445, 0.80), (0.535, 0.74)], fixed=True)),
         "2J049#1": dict(look="R", nohud=True), "2J049#2": dict(look="R", nohud=True), "2J050": dict(look="R", nohud=True)}
# 用户（看过 v10）："羽蛇神那一段单独演出不用处理原声，就按原声放出就行" → break 1 原样；break 2 背景压低（v4_audio 按峰值封顶定电平）
# 交接（用户 10-20："跟歌曲的衔接也很突兀，经常断断续续或者戛然而止又或者突然出现"）——真正重叠的等功率交叉：
#   song_fo 歌在停歌点前淡出的长度；xf_in 原声提前进来的长度（break 1 源从 0 s 起，提前不了）；nat_fi 原声淡入；
#   nat_tail 原声越过回歌点继续放的长度；nat_fo 原声淡出（覆盖 tail）；song_pre 歌提前淡入的长度（= 和原声尾巴交叉）
BREAKS = [dict(song=BREAK_SONG, runs=[("DE002", "DE038", 0.083, 43.70)], look=LOOK1, chapter=True, level=-13.0, audio="raw",
               song_fo=1.0, xf_in=0.083, nat_fi=0.5, nat_tail=0.6, nat_fo=1.2, song_pre=0.6),
          dict(song=BREAK2_SONG, runs=[("2J034", "2J050", 38.00, 74.60)], look=LOOK2, chapter=False, level=-14.0,
               song_fo=1.0, xf_in=1.0, nat_fi=1.0, nat_tail=0.5, nat_fo=1.4, song_pre=0.6)]
# ---------------------------------------------------------------- 台词演出（源时间；每行 (源起, 源止, 文字)；文字 = 两个 ASR 模型一致的部分，同音字按上下文：彗星 / 円卓 / 測って）
SPK = {"quetzalcoatl": ("QUETZALCOATL", "ケツァル・コアトル"), "ritsuka": ("RITSUKA FUJIMARU", "藤丸立香"),
       "bedivere": ("BEDIVERE", "ベディヴィエール"), "king": ("THE KING", "王"), "lion_king": ("LION KING", "獅子王")}
SPK_COL = {"ritsuka": ((96, 150, 236), (226, 232, 240)), "king": ((236, 206, 120), (120, 150, 220))}
DLINES = [
    dict(yid="deONLS7FcVI", spk="quetzalcoatl", side="R", rows=[(0.50, 3.00, "多くの命を絶滅させた"), (3.00, 5.88, "大衝突と力を見せてくれる!")]),
    dict(yid="deONLS7FcVI", spk="quetzalcoatl", side="R", rows=[(7.39, 9.35, "我が身を燃える岩となし"), (9.61, 11.69, "彗星となって大地を殺す!")]),
    dict(yid="deONLS7FcVI", spk="quetzalcoatl", side="R", shout=True, rows=[(12.41, 13.89, "ウルティモ・トペ"), (15.15, 16.69, "パターダ!")]),
    # 令咒发光的拳头 → 立香呐喊（"勝利を!" 正落在立香喊的镜头上）；前面 "令呪を…" 两个模型不一致，不上
    dict(yid="deONLS7FcVI", spk="ritsuka", side="L", rows=[(26.56, 29.02, "ケツァル・コアトルに勝利を!")]),
    dict(yid="deONLS7FcVI", spk="quetzalcoatl", side="R", shout=True, rows=[(34.54, 35.66, "燃えろ、闘魂!")]),
    # 贝德维尔器乐段（歌里无人声的 100.89–107.50）：宝具名
    dict(yid="_2j1V81wuR0", spk="bedivere", side="L", shout=True, rows=[(0.40, 2.76, "スイッチオン"), (2.76, 4.02, "アガートラム!")]),
    # break 2 回忆（男声 93–138 Hz = 贝德维尔；女声 190–250 Hz = 王）
    dict(yid="_2j1V81wuR0", spk="bedivere", side="L", chip="MEMORY", rows=[(38.36, 41.58, "なぜ 私のようなものを"), (41.58, 43.30, "円卓の騎士に選ばれたのです")]),
    dict(yid="_2j1V81wuR0", spk="king", side="R", chip="MEMORY", rows=[(43.84, 45.82, "他より劣っているから")]),
    dict(yid="_2j1V81wuR0", spk="bedivere", side="L", chip="MEMORY", rows=[(46.10, 46.90, "そんな…")]),
    dict(yid="_2j1V81wuR0", spk="king", side="R", chip="MEMORY", rows=[(47.44, 51.41, "王は 単純な強さ弱さで"), (51.41, 53.81, "人の繋がりを測ってはいけない")]),
    # 对峙（女声，狮子王）→ 特写口型同步（狮子王）
    dict(yid="_2j1V81wuR0", spk="lion_king", side="L", rows=[(64.13, 66.23, "なぜ、そこまで")]),     # 画面里狮子王在左（戴冠、毛领）
    dict(yid="_2j1V81wuR0", spk="lion_king", side="R", rows=[(69.11, 70.89, "そなたの王の命を"), (72.61, 73.53, "果たしたのだ")]),
]


def relook(blk, ch, **over):
    """沿用 v10/v11 的块，只改指定镜头的风格（去掉原来的 look / sfx / hud / nohud 再合并新的）。"""
    b = dict(blk, **over); its = []
    for it in blk["items"]:
        code, o = (it, {}) if isinstance(it, str) else (it[0], dict(it[1]))
        if code in ch:
            o = dict({k: v for k, v in o.items() if k not in ("look", "sfx", "hud", "nohud", "manga", "tone")}, **ch[code])
        its.append((code, o) if o else code)
    b["items"] = its
    return b


_E10 = {b["id"]: b for b in E10.BLOCKS}
_E11 = {b["id"]: b for b in E11.SONG_BLOCKS}
SONG_BLOCKS = [
    dict(_E11["V01"]),
    relook(_E10["V02"], {"O1006": dict(look="R", hud="viewfinder", who="伊什塔尔"),                 # 草原连环爆炸：取景器
                         "O1018": dict(look="M", manga="impact", tone="mono")}),                  # 城墙上飞踢：漫画冲击页
    dict(_E10["V03"]), dict(_E10["V04"]), dict(_E10["V05"]),
    relook(_E11["H1"], {"DL050": dict(look="R", hud="track", who=HIPPO)}),                        # 龟裂大地上的骑手：真实轨迹
    dict(_E11["V07"]),
    # ---- BREAK 1 ----
    # 78.45–88.25 武藏（用户："贝德维尔的片段可以往后放一些"）：歌在 IMPERSONATOR! 清唱处回来，先接武藏
    dict(_E10["V09"], t0=BREAK_SONG, t1=MUSASHI_END),
    # 88.25–107.50 贝德维尔：狮子王 → 银臂 / 圣枪 → 96.85 "And cast the first die!" = 圣枪宝具横幅 →
    #   100.89 器乐段：银臂 "スイッチオン…アガートラム"（原速，原声同步 + 喊招字幕）→ 骑士冲向城墙 → 红天闪电笼罩城堡 → 停歌（接 2J034）
    dict(id="BD", name="BEDIVERE", t0=MUSASHI_END, t1=BREAK2_SONG, kind="battle", enter=slit("bedivere"),
         items=[("2J005", dict(nohud=True, who="狮子王")), ("2J006", dict(hud="lock", who="狮子王")),
                ("2J013", dict(hud="track", who="贝德维尔")),                                     # 银臂挥击：轨迹
                ("2J019", dict(nohud=True)),
                ("2J026", dict(take=2.0, look="M", manga="impact", tone="mono", tag="CAMELOT  //  THE HOLY CITY")),
                ("2J027", dict(hud="type", who="狮子王")), ("2J029", dict(nohud=True)),
                ("2J030", dict(at=96.85, slam=12, burst=True, hud="np", who="狮子王")),          # 圣枪 Rhongomyniad
                ("2J031", dict(nohud=True)),
                ("2J001", dict(at=BD_GAP, s0=0.30, nohud=True, who="贝德维尔")), ("2J002", dict(nohud=True, who="贝德维尔")),
                ("2J008", dict(nohud=True)), ("2J032", dict(s0=36.09, nohud=True)), ("2J033", dict(nohud=True))]),
    # ---- BREAK 2 ----
    relook(_E10["V11"], {"QI015": dict(look="M", manga="impact", tone="mono")}, t0=BREAK2_SONG),   # 阿拉什从回歌处开始；怒吼拉弓：漫画冲击页
    dict(_E10["V12"]),
    # Saber Alter vs Rider：RV017 是固定机位的远景（左 Saber Alter、右 Rider，画面核对）→ 双人标注 link（手标两个目标；v10 的 VS 对决牌被评幼稚，已删）；
    #   RV019 源里是三个镜头、两人左右互换 → 对决牌对不准，改双色漫画分镜（用户："双人对打的漫画风格化处理可以改成双色"）
    relook(_E11["SA"], {"RV017": dict(look="R", hud="link", det=["manual"], manual=[1000, 520, 1330, 1300],
                                      vs=dict(keys=["artoria_alter", "medusa"], pts=[(0.455, 0.63, 0.15, 0.52), (0.76, 0.57, 0.11, 0.58)], fixed=True)),
                        "RV019": dict(look="M", manga="story"),
                        "RV101": dict(look="R", hud="np")}),                                      # 天马 / 黄金缰绳：宝具横幅（騎英の手綱）
    relook(_E11["AR"], {"PM075": dict(look="R", hud="track")}),                                   # 云海废墟间飞行：轨迹
    dict(_E11["V16"]),
]


# v10 第二版：新 HUD 模板分到这些镜头（单人、≥1.1 s；codex 要远侧有空、blueprint 要人物整身或主体清楚）
HUD_OVER = {"MU001": "codex", "PM066": "codex", "2J027": "codex", "OJ001": "codex",
            "O1014": "blueprint", "RV014": "blueprint", "PV001": "blueprint", "QI010": "blueprint"}


def char_times(yid, a, b, text):
    """一行字的逐字时刻（源时间）：ASR 词级时间按字数摊成"字时间轴"，正文按相对位置取（标点跟前一个字 +0.04 s）。"""
    ws = [w for w in DW[yid] if w[0] >= a - 0.06 and w[1] <= b + 0.06]
    tl = []
    for t0, t1, w in ws:
        w = re.sub(r"[、。!！?？\s…]", "", w)
        n = max(1, len(w))
        tl += [t0 + (t1 - t0) * i / n for i in range(n)]
    tl = tl or [a]
    PUN = "、。!！?？…・— "
    core = [ch for ch in text if ch not in PUN]
    out, k = [], 0
    for ch in text:
        if ch in PUN:
            out.append((round((out[-1][0] if out else tl[0]) + 0.04, 3), ch)); continue
        out.append((round(tl[min(len(tl) - 1, int(k * len(tl) / max(1, len(core))))], 3), ch)); k += 1
    return out


DW = json.load(open(os.path.join(ROOT, "data", "dialog_words.json"), encoding="utf-8"))


def song_part():
    E11.SONG_BLOCKS = SONG_BLOCKS                           # 排时 / 转场规则沿用 v11
    return E11.song_part()


def break_shots(B_, bi, f, native):
    """一次 break：源段按镜头原速铺开（SPLIT 里的镜头按源里的硬切 / 叠化点拆开），原声整段一条。返回 (shots, 末尾成片时间, 源→成片映射)。"""
    blk = dict(id="BR" if bi == 0 else "BS", name="INTERLUDE" if bi == 0 else "BEDIVERE II", kind="battle")
    out, maps, k = [], [], 0
    F_b0 = f
    for c0, c1, s0, s1 in B_["runs"]:
        t_run = f
        for code in run_codes(c0, c1):
            r = POOL[code]
            a_, b_ = max(r["in"], s0), min(r["out"], s1)
            cuts = [a_] + [x for x in SPLIT.get(code, []) if a_ < x < b_] + [b_]
            for j in range(len(cuts) - 1):
                key = code + ("#%d" % j if code in SPLIT else "")
                sh = native_shot(code, cuts[j], cuts[j + 1], f, blk, k, B_["look"].get(key, {})); k += 1
                sh["cine"] = True
                for x in ("det", "manual", "vs"):
                    if x in B_["look"].get(key, {}):
                        sh[x] = B_["look"][key][x]
                out.append(sh)
                f += cuts[j + 1] - cuts[j]
        xi = B_["xf_in"] if t_run == F_b0 else 0.0
        native.append(dict(t0=round(t_run - xi, 3), t1=round(f + B_["nat_tail"], 3), yid=POOL[c0]["yid"], src=round(s0 - xi, 3), level=B_["level"],
                           fi=B_["nat_fi"], fo=B_["nat_fo"], audio=B_.get("audio", "bg"),
                           note="break %d 原声（一刀不剪；%s；回歌后再放 %.1f s 交叉）" % (bi + 1, "原样" if B_.get("audio") == "raw" else "背景压低", B_["nat_tail"])))
        maps.append((POOL[c0]["yid"], s0, s1, t_run))
    return out, f, maps


def line_caps(maps, song_shots):
    """DLINES → 成片里的 line 字幕。源时间 → 成片：先找 break 原声段，再找原速同步的歌段镜头。"""
    def to_film(yid, x):
        for y, s0, s1, f0 in maps:
            if y == yid and s0 - 0.01 <= x <= s1 + 0.01:
                return f0 + (x - s0)
        for s in song_shots:
            if s["yid"] == yid and abs(s.get("speed", 1) - 1) < 0.05 and s["src_a"] - 0.01 <= x <= s["src_a"] + (s["b"] - s["a"]) * s["speed"] + 0.6:
                return s["a"] + (x - s["src_a"]) / s["speed"]
        raise ValueError((yid, x))
    caps = []
    for L in DLINES:
        rows = [[(round(to_film(L["yid"], tc), 3), ch) for tc, ch in char_times(L["yid"], a, b, txt)] for a, b, txt in L["rows"]]
        who, who_jp = SPK[L["spk"]]
        c = dict(style="line", rows=rows, who=who, who_jp=who_jp, key=L["spk"], side=L["side"], shout=L.get("shout", False))
        if L.get("chip"):
            c["chip"] = L["chip"]
        if L["spk"] in SPK_COL:
            c["col"] = [list(SPK_COL[L["spk"]][0]), list(SPK_COL[L["spk"]][1])]
        c["t0"] = round(rows[0][0][0] - 0.25, 3)
        c["t1"] = round(rows[-1][-1][0] + (0.8 if L.get("shout") else 1.1), 3)
        caps.append(c)
    for c in caps:                                          # 同侧下一句出现前 0.2 s 收掉；也不拖过所在 break 的结尾
        nxt = [d["t0"] for d in caps if d["side"] == c["side"] and d["t0"] > c["t0"]]
        if nxt:
            c["t1"] = round(min(c["t1"], min(nxt) + 0.05), 3)
        for y, s0, s1, f0 in maps:
            if f0 <= c["t0"] <= f0 + (s1 - s0):
                c["t1"] = round(min(c["t1"], f0 + (s1 - s0) + 0.4), 3)
    return caps


def build():
    song_shots, song_trans, mus = song_part()
    for sh in song_shots:
        if sh["code"] in HUD_OVER and sh["look"] in ("R", "P") and not sh.get("nohud"):
            sh["hud"] = HUD_OVER[sh["code"]]
    # ---- 序章（绫香与 Richard 初识，同 v11）
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
    f_line = lambda x: t_r1 + (x - s_r1)
    caps.append(dict(t0=round(f_line(LINE[0]) - 0.05, 3), t1=round(f_line(LINE[1]) + 0.6, 3), text=LINE[2], style="serif"))
    # 用户（看过 v10 第四版）："intro 绫香和理查那一段的原声音乐也去掉吧，仅保留环境音（如有）和清晰的角色语音"
    #   → 序章 / 标题卡下原来垫的 11vAh33_2eI（实测是马修那场的配乐）和 Richard 那场的整轨原声（带配乐）都去掉；
    #     Richard 的台词用 MDX 分离干声（同 v7 做法）；这几个源去掉配乐后只剩乐音残渣，没有能用的环境声
    # 再改（用户："开头完全去掉素材原声的音乐又显得人物干声很粗糙，但是保留音乐会和歌曲衔接有些突兀，你想想解决方案"）：
    #   实测 Richard 这场原声里分离模型认不出配乐（去配乐前后一样），粗糙来自分离干声 → 改回这场的原声（人声自带空间感）；
    #   台词说完后原声一边低通扫走一边长出混响，尾巴衰减到歌进来之后（v4_audio.recede）；歌的入场加反向混响 + 低音渐入（song_intro）
    native.append(dict(t0=round(t_r1 - 0.5, 3), t1=round(P_END + 1.4, 3), yid="EpLKseoUV6Q", src=round(s_r1 - 0.5, 3), level=-15.0, fi=0.4, fo=1.2,
                       bg_db=0.0, recede=dict(t=round(f_line(LINE[1]) + 0.35, 3), f0=12000.0, f1=500.0, rt60=3.8, wet=0.9),
                       note="Richard 初见原声（汝が俺のマスターか）→ 台词后退远 + 混响尾巴接歌"))
    SONG0 = P_END + TITLE_PRE
    # ---- 歌段 + break（成片时间）
    segs, brk, binfo, maps, f, prev = [], [], [], [], SONG0, 0.0
    for bi, B_ in enumerate(BREAKS):
        segs.append((f, prev, B_["song"]))
        f += B_["song"] - prev
        F_b0 = f
        sh_, f, mp = break_shots(B_, bi, f, native)
        brk += sh_; maps += mp
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
    # ---- 标题卡
    TI = dict(id="TI", name="TITLE", kind="intro")
    e0 = next(s for s in song_shots if s["id"] == "E01_01")
    code, s0, s1 = TITLE_SRC
    r = POOL[code]
    it = dict(code=code, r=r, s0=s0, s1=s1, lo=r["in"], hi=r["out"], o=dict(look="T"), a=P_END, b=e0["a"], speed=(s1 - s0) / (e0["a"] - P_END))
    title = shot_of(it, TI, 0)
    title["src_a"], title["lo"], title["hi"] = s0, s0, s1 - 1 / 60.0   # 接着序章那一帧往下放（不加 shot_of 的转场余量）
    title["title_spec"] = dict(kicker="FATE SERIES  //  AMV", title="PROVANT", sub="SawanoHiroyuki[nZk]:Jean-Ken Johnny", exit=False, fade_in=14, delay=12)
    # 标题卡下不垫声音（原来的"夜风"实测是配乐），歌从标题卡下进来
    # ---- 贝德维尔器乐段：银臂 "スイッチオン…アガートラム"（2J001 / 2J002 原速，原声同步；歌里这段没有人声）
    g1 = next(s for s in song_shots if s["code"] == "2J001")
    g_end = g1["a"] + (4.60 - g1["src_a"]) / g1["speed"]                 # "アガートラム" 4.02 喊完，余音再放 0.45 s 淡出（不戛然而止）
    native.append(dict(t0=round(g1["a"], 3), t1=round(g_end, 3), yid=g1["yid"], src=round(g1["src_a"], 3), level=-13.0, fi=0.12, fo=0.6, bg_db=-15.0,
                       note="贝德维尔：スイッチオン…アガートラム（器乐段，去配乐原声同步）"))
    print("  器乐段银臂：成片 %.2f–%.2f（歌 %.2f 起），速度 %.3f" % (g1["a"], g_end, g1["sa"], g1["speed"]))
    # ---- 台词（成片时间）：武藏那句 v9 在器乐段，现在武藏挪到唱段里 → 不放（不在人声上叠台词）
    dialog = [dict(d, t=round(s2f(d["t"]), 3)) for d in V9.DIALOG if d["yid"] != "MUkengo4K60"]
    for yid, s0_, s1_, who_, note_ in OKITA_LINES:
        sh = next(x for x in song_shots if x["yid"] == yid and x["src_a"] - 0.02 <= s0_ <= x["src_a"] + (x["sb"] - x["sa"]) * x["speed"] + 0.02)
        dialog.append(dict(t=round(sh["a"] + (s0_ - sh["src_a"]) / sh["speed"], 3), yid=yid, s0=s0_, s1=s1_, who=who_, note=note_))
    F_SONG_END = s2f(SONG_END, end=True)
    t4 = s2f(170.40)
    dialog.append(dict(t=round(t4, 3), yid="Cf9uMIffDTQ", s0=MASH_TAIL[0][0], s1=MASH_TAIL[0][1], who="玛修", note="歌结束后：そういうもののために"))
    t5 = t4 + (MASH_TAIL[0][1] - MASH_TAIL[0][0]) + 0.22
    dialog.append(dict(t=round(t5, 3), yid="Cf9uMIffDTQ", s0=MASH_TAIL[1][0], s1=MASH_TAIL[1][1], who="玛修", note="私たちは戦ってきたのですから"))
    T1 = round(t5 + (MASH_TAIL[1][1] - MASH_TAIL[1][0]) + 1.2 + 3.0, 3)
    for d in dialog:
        if d["yid"] == "Cf9uMIffDTQ":                       # 贞德（开场）/ 马修（片尾）：v7 的 MDX 分离干声
            d["voice"] = "sep"
    dialog.sort(key=lambda d: d["t"])
    # ---- 尾声：马修 Cs
    OU = dict(id="E12", name="MASH", kind="outro")
    a_cs = s2f(165.5815)
    r = POOL["11065"]
    it = dict(code="11065", r=r, s0=96.10, s1=105.70, lo=r["in"], hi=r["out"], o=dict(look="Cs"), a=a_cs, b=T1, speed=(105.70 - 96.10) / (T1 - a_cs))
    cs = shot_of(it, OU, 0); cs["id"] = "E12_00"; cs["seg"] = "E12"
    # 片尾原来垫的"环境音"（11vAh33_2eI @99）实测是马修那场的配乐（去掉配乐后只剩乐音残渣）→ 用户要马修"去掉原声的音乐，用清晰的人声"，不再垫
    song_shots = [s for s in song_shots if s["id"] != "E12_00"]
    shots = sorted(pro + [title] + song_shots + brk + [cs], key=lambda s: s["a"])
    for kx in range(1, len(shots)):
        if abs(shots[kx]["a"] - shots[kx - 1]["b"]) > 2e-3:
            print("  ! 成片不连续 %s %.3f → %s %.3f" % (shots[kx - 1]["id"], shots[kx - 1]["b"], shots[kx]["id"], shots[kx]["a"]))
    V9.assign_hud([s for s in shots if s["look"] in ("R", "P") and not s.get("nohud")])
    # ---- 转场
    trans = list(song_trans)
    for sh in pro[1:]:
        trans.append(dict(t=sh["a"], kind="gcut", pre=2, post=3)) if sh["code"] not in ("EP126",) else None
    # 序章 → 标题卡是同一镜头的速度斜坡：不加转场
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
    for bi, b in enumerate(binfo):
        glitch += [(round(b["F0"] - 0.30, 3), round(b["F0"], 3), 0.1, 0.55), (round(b["F0"], 3), round(b["F0"] + 0.14, 3), 0.9, 0.0),
                   (round(b["F1"] - 0.35, 3), round(b["F1"], 3), 0.12, 0.65), (round(b["F1"], 3), round(b["F1"] + 0.14, 3), 1.0, 0.0)]
        caps.append(dict(t0=round(b["F0"] + 0.3, 3), t1=round(b["F1"] + 1.3, 3), style="player", title="PROVANT  ―  SawanoHiroyuki[nZk]",
                         resume=round(b["F1"], 3), song_t=b["song"], song_len=SONG_END, label_pause="PAUSED  ·  SOURCE AUDIO", label_play="NOW PLAYING"))
        if b.get("chapter"):
            caps.append(dict(t0=round(b["F0"] + 0.9, 3), t1=round(b["F0"] + 4.8, 3), style="chapter", kicker="INTERLUDE  //  QUETZALCOATL  ·  RIDER",
                             text="ULTIMO TOPE PATADA", sub="ケツァル・コアトル  ―  絶対魔獣戦線バビロニア"))
    cine = []
    for bi, b in enumerate(binfo):                          # 遮幅：break 1 等 INTERLUDE 缝转场打开后再收拢；break 2 从停歌处收拢；回歌前打开
        a0 = b["F0"] + (0.30 if bi == 0 else 0.0)
        cine.append(dict(t0=round(b["F0"], 3), t1=round(b["F1"], 3), bar_in=[round(a0, 3), round(a0 + CINE_IN, 3)],
                         bar_out=[round(b["F1"] - CINE_OUT, 3), round(b["F1"], 3)]))
    lc = line_caps(maps, song_shots)
    caps += lc
    punch = [round(s2f(x), 3) for x in E10.PUNCH]
    cfg = dict(T0=0.0, T1=T1, out="PROVANT_AMV_v10_video.mp4", shots=shots, trans=trans, lyrics=lyr, glitch=glitch, punch=punch,
               fade=[(round(T1 - 3.0, 3), T1)], dialog=dialog, hud_v=5, song_map=song_map, native_audio=native, captions=caps, cine=cine,
               song_xf=[dict(fo=BREAKS[0]["song_fo"], intro=dict(hp_hz=300.0, hp_s=0.7, swell_s=1.8, swell_db=-10.0))] +
                       [dict(pre=BREAKS[k]["song_pre"], fo=(BREAKS[k + 1]["song_fo"] if k + 1 < len(BREAKS) else 0.0)) for k in range(len(BREAKS))])
    fw = V9.flash_check(json.loads(json.dumps(shots)))
    fw = [w for w in fw if not w[1].startswith("slide")]
    dd = V9.dupes(shots, {s["code"].split(">")[0] for s in shots})
    for s in shots:
        s.pop("_risky", None)
    json.dump(cfg, open(os.path.join(ROOT, "data", "edl_v12.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # ---- 统计 + 人看的表
    n = len(shots)
    lk = Counter(s["look"] + ("-" if s.get("nohud") else "") for s in shots)
    hk = Counter(s.get("hud") for s in shots if s["look"] in ("R", "P") and not s.get("nohud"))
    gpt = Counter()
    for s in shots:
        gpt[s["group"]] += s["b"] - s["a"]
    Ls = ["# EDL v12（自动生成：scripts/edl_v12.py）", "",
          "成片 %.2f s；序章 0–%.2f（无歌）→ 标题 → 歌 %.2f 起；break %s；歌结束 %.2f；淡出到 %.2f" % (
              T1, P_END, SONG0, "、".join("%.2f–%.2f（%.1f s）" % (b["F0"], b["F1"], b["F1"] - b["F0"]) for b in binfo), F_SONG_END, T1),
          "镜头 %d；风格 %s；HUD %s；转场 %s；系列时长 %s" % (n, dict(lk), dict(hk), dict(Counter(x["kind"] for x in trans)), {k_: round(v, 1) for k_, v in gpt.items()}), ""]
    cur = None
    for s in shots:
        if s["block"] != cur:
            cur = s["block"]; Ls += ["", "## %s" % cur]
        Ls.append("- `%s` %7.3f–%7.3f %-3s%s %-10s x%.2f %s/%s %s — %s（%s）" % (
            s["id"], s["a"], s["b"], s["look"], "-" if s.get("nohud") else (s.get("hud", "") and ":" + s["hud"]), s["code"], s.get("speed", 1),
            s["group"], s["res"], s["kind"], s["what"][:40], s["who"][:16]))
    Ls += ["", "## 台词（配音）"] + ["- %.2f %s %s" % (d["t"], d["who"], d.get("note", "")) for d in dialog]
    Ls += ["", "## 台词演出（字幕）"] + ["- %.2f–%.2f %s %s：%s" % (c["t0"], c["t1"], c["side"], c["who"], " / ".join("".join(ch for _, ch in r_) for r_ in c["rows"])) for c in lc]
    Ls += ["", "## 原声"] + ["- %.2f–%.2f %s @%.2f %s" % (x["t0"], x["t1"], x["yid"], x["src"], x["note"]) for x in native]
    Ls += ["", "## 预检提示"] + ["- " + str(w) for w in fw] + ["- 画面重复：" + str(dd)]
    open(os.path.join(ROOT, "EDL_v12.md"), "w", encoding="utf-8").write("\n".join(Ls))
    print(Ls[2]); print(Ls[3])
    return cfg


def sheets(cfg):
    from cards import src_path
    from PIL import Image, ImageDraw, ImageFont
    out = os.path.join(ROOT, "ae", "qa", "v12"); os.makedirs(out, exist_ok=True)
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
            d.text(((j % cols) * 320 + 4, (j // cols) * 210 + 181), "%s %s%s %s %.2f %s" % (s["id"][-2:], s["look"], "-" if s.get("nohud") else (":" + s.get("hud", "") if s.get("hud") else ""),
                                                                                       s["code"], s["a"], tr_at.get(round(s["a"], 3), "|")), font=fnt, fill=(255, 220, 90))
        sheet.save(os.path.join(out, "%02d_%s.jpg" % (k, re.sub(r"\W", "_", blk))), quality=85)
    print("sheets ->", out)


if __name__ == "__main__":
    cfg = build()
    if "--no-sheets" not in sys.argv:
        sheets(cfg)

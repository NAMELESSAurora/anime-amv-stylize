# -*- coding: utf-8 -*-
"""
v2 成片的镜头池：data/shots/<yid>.json（切镜 + 指标）+ data/shots/tags/<yid>.json（子代理标签）
+ data/shots/cg3d_audit.json（旧池 3D 复查）→ data/shot_pool_v2.json。

用户 10-18 的取舍：
  · 不要 3D 游戏素材 → tags 里 cg3d=true 的、3D 复查列出的、FGO Arcade OP（kxz8k4ycXCA）整条，一律剔除
  · 素材尽量 2K 及以上 → 每镜带 res（有 1440 原帧率版 = 1440）
  · "多用一些 FATE STAY NIGHT 剧场版以及 FATE STRANGE FAKE 和 FGO 的素材" → prio：HF 剧场版 1.3 / FSF 1.2 / FGO 1.0 /
    UBW（ufotable）0.6 / Apocrypha 0.6（贞德另加分）/ HF 游戏 OP 0.4 / hollow ataraxia、Samurai Remnant 0
  · 卡尔纳 vs 齐格飞（RxmNey6OU_s）右下角水印：用户允许裁掉 → 用 note 里"裁掉水印后≈X"的等级，crop=1.05（绕顶边中点放大，
    水印 y 1385–1415 被推出画面）；S110 起是上传者拼的别的番，不用
  · FSF OP 用 4K 的 5Jyflcob9_k（_clean 版 88.5 s 之后没有画面），1080 的同一段 OP（tcC9g9dlv58）不再用
  · 吉尔伽美什：用户不喜欢 → gil=True，选镜时重罚（不当主角）
剔除：X 级、带文字（text≠none）、同一画面的重复段。
"""
import glob, json, os, re

ROOT = r"D:\A\provant"
SH = os.path.join(ROOT, "data", "shots")

AB = {
    # FSF
    "FS": "5Jyflcob9_k", "EP": "EpLKseoUV6Q", "DL": "DlyEbc5AMUw", "RH": "rHlLzmlc69I", "TC": "tcC9g9dlv58",
    # FGO
    "ON": "ONuRaCcA8iw", "YU": "Yuq57lj3LMU", "62": "62hOiNH0K44", "QI": "qiANLuvPb9o", "2J": "_2j1V81wuR0",
    "O1": "o1H-NNBjHbE", "DE": "deONLS7FcVI", "PM": "PmOY1wzlRjo", "11": "11vAh33_2eI", "L0": "L02aPFAhTCk",
    "UU": "UuxL121gwN8", "G9": "G9nHP6aRsjE", "7R": "7R9fdZLTYjo", "CF": "Cf9uMIffDTQ", "KX": "kxz8k4ycXCA",
    # HF 剧场版
    "SB": "fKoHcAM0PEk", "LA": "5IiGdXV-wow", "RV": "tlKnVXKaZYs", "RS": "jqQkurxOj-Q", "ME": "vUqzHyadZAA",
    "AA": "vSaILO_7UVk", "RA": "a8QQN9RDrSk", "SA": "B5DdMAhWCsE", "GS": "aT7kG7zdkak", "NL": "9c4ozWgw3Ns",
    "RN": "3U3DFMZBddA", "SK": "rZOxWc1nGt8",
    "OJ": "OKjiend4K60", "OY": "OKyamatai4K", "OA": "OKAteito4K6", "MU": "MUkengo4K60",   # 10-19 用户给的冲田 / 武藏 CM
    "KK": "oZlRdc9gxE4", "KA": "cpxwZP1GfAs", "RF": "hQcHTLb-FJs", "RD": "tY4SZ_NKKb4", "BT": "uvLyRoYugbc", "PR": "ghwoaF-VYS0",
    # Apocrypha
    "A1": "CQ8v15KMND4", "A2": "VobxMYXmkao", "AE": "5m13-yHL7EU", "AK": "RxmNey6OU_s", "AS": "BWiA7y_XNHY",
    "AT": "kgqkkGuyRjU",
    # UBW / HF 游戏 OP / 其他
    "UB": "fh57udNMKvU", "UV": "ViVJDXIBcMk", "UR": "g3n5c5CwDzs", "HV": "5ZLnndSF3rk", "H2": "iWpZhkxdcss",
    "HA": "7CBcMPi6Q84", "HM": "Mmz4LBlaR8E", "SR": "3zV3RgFcyiY",
}
YID2AB = {v: k for k, v in AB.items()}
GROUP = {}
for k in ("FS", "EP", "DL", "RH", "TC"):
    GROUP[AB[k]] = "FSF"
for k in ("ON", "YU", "62", "QI", "2J", "O1", "DE", "PM", "11", "L0", "UU", "G9", "7R", "CF", "KX", "OJ", "OY", "OA", "MU"):
    GROUP[AB[k]] = "FGO"
for k in ("SB", "LA", "RV", "RS", "ME", "AA", "RA", "SA", "GS", "NL", "RN", "SK", "KK", "KA", "RF", "RD", "BT", "PR"):
    GROUP[AB[k]] = "HFM"
for k in ("A1", "A2", "AE", "AK", "AS", "AT"):
    GROUP[AB[k]] = "APO"
for k in ("UB", "UV", "UR"):
    GROUP[AB[k]] = "UBW"
for k in ("HV", "H2"):
    GROUP[AB[k]] = "HFG"
for k in ("HA", "HM", "SR"):
    GROUP[AB[k]] = "OTH"
PRIO = {"HFM": 1.3, "FSF": 1.2, "FGO": 1.0, "UBW": 0.6, "APO": 0.6, "HFG": 0.4, "OTH": 0.0}
SKIP_SRC = {"kxz8k4ycXCA", "tcC9g9dlv58"}
# 用户 10-18："stay night 的剧场版就用 D:\Fate_AMV\hf 这个的素材，其他的 STAY NIGHT 的素材比如老剧场的就不要用了"
# → HF / UBW / HF 游戏 OP / hollow ataraxia 这些 stay night 系素材，只留文件夹里有的那些 ID
HF_DIR = r"D:\Fate_AMV\hf"
HF_ALLOWED = set(re.findall(r"\[([A-Za-z0-9_\-]{11})\]", " ".join(os.listdir(HF_DIR)))) if os.path.isdir(HF_DIR) else set()
FSN_GROUPS = {"HFM", "UBW", "HFG"}
FSN_EXTRA = {"7CBcMPi6Q84", "Mmz4LBlaR8E"}          # hollow ataraxia（stay night 续作）也算
DUP_RANGES = [("aT7kG7zdkak", 36, 43), ("jqQkurxOj-Q", 1, 13)]
CROP = {"RxmNey6OU_s": (1.05, "top"),        # 右下角水印 PREZTIYU：绕顶边中点放大 5%
        "oZlRdc9gxE4": (1.11, "bottom")}    # 左上角水印 Sr.Nakamura（x 2–22%, y 4–9%）：绕底边中点放大 11%，仍约 2304×1296
LIMIT = {"5Jyflcob9_k": 88.4, "RxmNey6OU_s": 274.0}
FAV = re.compile(r"贞德|阿尔托莉雅|Saber|狮心王|理查|Richard|冲田|Artoria|阿尔托利亚|武藏")


def ytid(name):
    m = re.findall(r"__([A-Za-z0-9_\-]{11})\.mov$", name)
    return m[-1] if m else None


def res_map():
    have = {}
    for sub in ("conform_1440", "conform_1440_new"):
        d = os.path.join(ROOT, "media", sub)
        for f in os.listdir(d) if os.path.isdir(d) else []:
            if f.lower().endswith(".mov") and ytid(f):
                have[ytid(f)] = os.path.join(d, f)
    return have


def main():
    hi = res_map()
    audit = json.load(open(os.path.join(SH, "cg3d_audit.json"), encoding="utf-8"))
    cg = {(a["yid"], a["n"]) for a in audit["cg3d"]}
    pool = []
    for y, code in sorted(YID2AB.items(), key=lambda kv: kv[1]):
        sp, tp = os.path.join(SH, y + ".json"), os.path.join(SH, "tags", y + ".json")
        if not (os.path.exists(sp) and os.path.exists(tp)):
            print("缺", code, y); continue
        src = json.load(open(sp, encoding="utf-8"))
        tags = {t["n"]: t for t in json.load(open(tp, encoding="utf-8"))["shots"]}
        for s in src["shots"]:
            t = tags.get(s["n"], {})
            r = dict(s)
            r.update(yid=y, code="%s%03d" % (code, s["n"]), group=GROUP[y], prio=PRIO[GROUP[y]],
                     res=1440 if y in hi else 1080, grade=t.get("grade", "?"), kind=t.get("kind", ""), mood=t.get("mood", ""),
                     who=t.get("who", ""), what=t.get("what", ""), text=t.get("text", "none"), note=t.get("note", ""),
                     cg3d=bool(t.get("cg3d")) or (y, s["n"]) in cg)
            if y in CROP:
                m = re.search(r"裁掉水印后≈([ABC])", r["note"])
                if m and r["text"] in ("none", "", None) or (m and "水印" in str(r["text"])):
                    r["grade"], r["text"] = m.group(1), "none"
                r["crop"], r["crop_anchor"] = CROP[y]
            if t.get("crop"):                                # 逐镜裁切（CM 的制作字样 / 台标）
                r["crop"], r["crop_anchor"] = t["crop"]
            r["gil"] = "吉尔伽美什" in r["who"]
            r["fav"] = bool(FAV.search(r["who"]))
            r["split"] = bool(re.search(r"叠化|划像|多段|拆|溶|dissolve|wipe|→", r["note"] + r["what"]))
            dup = any(y == y0 and a <= s["n"] <= b for y0, a, b in DUP_RANGES) or bool(t.get("dup_of"))
            fsn_out = (GROUP[y] in FSN_GROUPS or y in FSN_EXTRA) and y not in HF_ALLOWED
            over = y in LIMIT and s["out"] > LIMIT[y]
            r["usable"] = (r["grade"] in ("A", "B", "C") and r["text"] in ("none", "", None) and not r["cg3d"]
                           and y not in SKIP_SRC and not dup and not over and not fsn_out)
            pool.append(r)
    use = [r for r in pool if r["usable"]]
    json.dump(use, open(os.path.join(ROOT, "data", "shot_pool_v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    from collections import Counter
    c = Counter((r["group"], r["res"]) for r in use)
    print("全部 %d，可用 %d" % (len(pool), len(use)))
    for g in ("HFM", "FSF", "FGO", "APO", "UBW", "HFG", "OTH"):
        a = [r for r in use if r["group"] == g]
        print("  %-4s %4d 镜（1440 %4d / 1080 %4d），A %d" % (g, len(a), c[(g, 1440)], c[(g, 1080)], sum(r["grade"] == "A" for r in a)))
    print("  剔除 3D %d，吉尔伽美什镜头 %d（可用池里，重罚）" % (sum(r["cg3d"] for r in pool), sum(r["gil"] for r in use)))


if __name__ == "__main__":
    main()

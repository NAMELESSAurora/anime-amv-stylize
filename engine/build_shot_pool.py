# -*- coding: utf-8 -*-
"""
把镜头数据（data/shots/<yid>.json）和子代理打的标签（data/shots/tags/<yid>.json）合并成一个镜头池，
供 MV Plan 选镜头用。

规则：
  - X 级、带文字（text != none）的镜头剔除
  - 已知问题源：5Jyflcob9_k 与 tcC9g9dlv58 是同一段 OP，前者结尾带 Anicrad 水印 → 默认只用 tcC9g9dlv58
  - 标签 note 里提到「叠化 / 划像 / 多段」的长镜头标 split=True，选用前要拆
  - dark ≥ 0.5 的镜头标 dark=True（漫画化后大概率整块墨黑，见 MANGA_CHAIN_NOTES §七）

输出:
  data/shot_pool.json   每个可用镜头一条（含 yid/series/n/in/out/dur/grade/kind/mood/who/what/指标/标志）
  data/SHOT_POOL.md     按 系列 × 情绪 × 时长 汇总，以及每条源的 A 级镜头清单（人看的版本）

用法:
  python build_shot_pool.py
"""
import glob, json, os, re, sys
from collections import Counter, defaultdict

ROOT = r"D:\A\provant"
SHOTS = os.path.join(ROOT, "data", "shots")
DUP_SKIP = {"5Jyflcob9_k"}           # 与 tcC9g9dlv58 同一段 OP，带水印
# 同一段画面在两条源里都有：(源, 起镜号, 止镜号) 这段跳过，留另一条（HF-2 标注报告：黑樱变身）
DUP_RANGES = [("aT7kG7zdkak", 36, 43),   # = rZOxWc1nGt8 S016–019
              ("jqQkurxOj-Q", 1, 13)]   # = tlKnVXKaZYs S079–090（HF-1 标注报告）
SPLIT_RE = re.compile(r"叠化|划像|多段|拆|溶|dissolve|wipe", re.I)
BAR = 2.3303


def load():
    pool, missing = [], []
    for f in sorted(glob.glob(os.path.join(SHOTS, "*.json"))):
        yid = os.path.splitext(os.path.basename(f))[0]
        if yid in ("index",):
            continue
        src = json.load(open(f, encoding="utf-8"))
        tp = os.path.join(SHOTS, "tags", yid + ".json")
        if not os.path.exists(tp):
            missing.append(yid)
            continue
        tags = {t["n"]: t for t in json.load(open(tp, encoding="utf-8"))["shots"]}
        for s in src["shots"]:
            t = tags.get(s["n"], {})
            rec = dict(s)
            rec.update({
                "yid": yid, "series": src["series"],
                "grade": t.get("grade", "?"), "kind": t.get("kind", ""), "mood": t.get("mood", ""),
                "who": t.get("who", ""), "what": t.get("what", ""), "text": t.get("text", "none"),
                "note": t.get("note", ""),
            })
            rec["split"] = bool(SPLIT_RE.search(rec["note"]))
            rec["dark_flag"] = s["dark"] >= 0.5
            dup = any(yid == y0 and a <= s["n"] <= b for y0, a, b in DUP_RANGES)
            rec["usable"] = (rec["grade"] in ("A", "B", "C") and rec["text"] in ("none", "", None)
                             and yid not in DUP_SKIP and not dup)
            pool.append(rec)
    return pool, missing


def dur_bin(d):
    if d < 1.0: return "<1s"
    if d < BAR: return "1s–1小节"
    if d < 2 * BAR: return "1–2小节"
    return "≥2小节"


def main():
    pool, missing = load()
    use = [r for r in pool if r["usable"]]
    json.dump(use, open(os.path.join(ROOT, "data", "shot_pool.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    L = ["# 镜头池汇总（自动生成：scripts/build_shot_pool.py）", ""]
    L.append("全部镜头 %d，可用 %d（剔除 X 级、带文字、重复源 %s）。" %
             (len(pool), len(use), "/".join(sorted(DUP_SKIP))))
    if missing:
        L.append("**尚未打标签、未计入的源：** " + ", ".join(missing))
    L += ["", "## 按系列 × 等级", "", "| 系列 | A | B | C | 合计 | 其中 ≥1 小节 | 偏暗 | 需拆分 |", "|---|---|---|---|---|---|---|---|"]
    for ser in ("SF", "FGO", "HF"):
        rs = [r for r in use if r["series"] == ser]
        g = Counter(r["grade"] for r in rs)
        L.append("| %s | %d | %d | %d | %d | %d | %d | %d |" % (
            ser, g["A"], g["B"], g["C"], len(rs), sum(r["dur"] >= BAR for r in rs),
            sum(r["dark_flag"] for r in rs), sum(r["split"] for r in rs)))

    L += ["", "## A/B 级按情绪 × 时长", "", "| 情绪 | <1s | 1s–1小节 | 1–2小节 | ≥2小节 |", "|---|---|---|---|---|"]
    tab = defaultdict(Counter)
    for r in use:
        if r["grade"] in ("A", "B"):
            tab[r["mood"] or "?"][dur_bin(r["dur"])] += 1
    for m in sorted(tab, key=lambda k: -sum(tab[k].values())):
        L.append("| %s | %d | %d | %d | %d |" % (m, tab[m]["<1s"], tab[m]["1s–1小节"], tab[m]["1–2小节"], tab[m]["≥2小节"]))

    L += ["", "## 各源 A 级镜头", ""]
    by = defaultdict(list)
    for r in use:
        if r["grade"] == "A":
            by[(r["series"], r["yid"])].append(r)
    for (ser, yid) in sorted(by):
        L.append("### %s %s" % (ser, yid))
        for r in by[(ser, yid)]:
            flags = "".join([" ⚠暗" if r["dark_flag"] else "", " ⚠拆" if r["split"] else ""])
            L.append("- S%03d `%s` +%.1fs %s/%s — %s（%s）%s" % (
                r["n"], "%d:%05.2f" % (int(r["in"] // 60), r["in"] % 60), r["dur"], r["kind"], r["mood"],
                r["what"], r["who"], flags))
        L.append("")
    open(os.path.join(ROOT, "data", "SHOT_POOL.md"), "w", encoding="utf-8").write("\n".join(L))
    print("pool %d / usable %d / missing tags: %s" % (len(pool), len(use), missing or "none"))


if __name__ == "__main__":
    main()

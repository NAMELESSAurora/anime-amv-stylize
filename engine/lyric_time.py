# -*- coding: utf-8 -*-
"""
逐词起点（用户 10-19 第 1 条："歌词跟拍需要优化一下"）。
v2 的做法是句长均分后吸附八分音符网格——和人声实际咬字对不上。现在：
  1. 歌曲干声：scripts/sep_mdx.py（UVR MDX Voc_FT）分出的 E:\\provant_work\\audio\\provant_vocals.wav
  2. 干声起音包络（150–5000 Hz，hop 128 @22.05k ≈ 5.8 ms），peak_pick 出候选起音
  3. 每句：词数 N、每词音节估计（英文元音组 / 日文假名 1、汉字 1.6、小假名 0）→ 期望起点；
     第 1 个词在句首 ±0.18 s 里找最强起音；其余词用动态规划在候选起音里选一条单调路径，
     得分 = 起音强度 − 2.2 × |偏离期望| / 平均词长，相邻词 ≥ 0.09 s；没有合适起音就用期望时间（扣分）。
输出 data/lyric_words.json：{句号: [[词, 起点秒], ...]}；v2_preview.burst_times 优先读它。
画面上词比咬字提前 1 帧出现（视觉领先，读起来是"打在拍上"）。
"""
import json, os, re, sys
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(__file__))
ROOT = r"D:\A\provant"
VOC = r"E:\provant_work\audio\provant_vocals.wav"
OUT = os.path.join(ROOT, "data", "lyric_words.json")
SMALL = set("ゃゅょっぁぃぅぇぉャュョッァィゥェォー")


def syl(w):
    if re.search(r"[A-Za-z]", w):
        n = 0
        for t in re.findall(r"[a-z']+", w.lower()):
            g = len(re.findall(r"[aeiouy]+", t))
            if t.endswith("e") and not t.endswith(("le", "ee")) and g > 1:
                g -= 1                                   # 词尾不发音的 e：take / some / face
            if t.endswith("ed") and not t.endswith(("ted", "ded")) and g > 1:
                g -= 1
            n += max(1, g)
        return max(1, n)
    n = 0.0
    for ch in w:
        if ch in SMALL:
            continue
        n += 1.6 if 0x4E00 <= ord(ch) <= 0x9FFF else 1.0
    return max(1.0, n)


# 手工校正（已含 1 帧视觉领先）。12 句是拖长音慢唱，一小节一个词（用户 10-19："还是抢拍，每个单词大概 2S 时间"）；
# 字幕时间轴把整句压在 28.1–31.0，实际唱到 38.2（干声音高：JUST 28.15 C4→D4 长音 / LIKE 30.5 F4 强起音 / ALL 32.6 /
# THE 35.0 G4 / REST? 37.4 G4 → 37.7 跳 G5 上扬 = 问句）。END 覆盖整句的显示结束时间。
# v7 更正（用户："单词出的太慢了，每个词差不多 1.25 秒左右，rest 尾音稍微长了一点"）：按干声起音峰（27.95 / 29.19 / 30.51 / 31.71 / 32.68）
MANUAL = {12: [("JUST", 28.13), ("LIKE", 29.18), ("ALL", 30.48), ("THE", 31.68), ("REST?", 32.66)]}
END = {12: 34.60}


def main():
    import v2_preview as P
    y, sr = librosa.load(VOC, sr=22050, mono=True)
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmin=150, fmax=5000, aggregate=np.median)
    env = env / (np.percentile(env, 99) + 1e-9)
    tt = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    pk = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=12, post_avg=12, delta=0.08, wait=int(0.07 * sr / hop))
    on_t, on_s = tt[pk], env[pk]
    out, rep = {}, []
    for i, r in enumerate(P.LYR):
        s0, e0 = r["start"] / 1000.0, r["end"] / 1000.0
        words = P.burst_words(i)
        n = len(words)
        sy = np.array([syl(w) for w in words], float)
        dur = max(0.4, e0 - s0)
        exp = s0 + 0.92 * dur * np.concatenate([[0], np.cumsum(sy)[:-1]]) / sy.sum()
        avg = dur / n
        m = (on_t > s0 - 0.15) & (on_t < e0 + 0.3)
        cand = [(float(t), float(s)) for t, s in zip(on_t[m], on_s[m])]
        # DP（含第 1 个词）：状态 = 上一个选中的候选下标；空缺 = 用期望时间
        states = {}
        for j, (t, sc_) in enumerate(cand):                 # 第 1 个词：句首 -0.15…+0.35 s
            if -0.15 <= t - s0 <= 0.35:
                states[j] = (1.5 * sc_ ** 0.7 - 1.0 * abs(t - s0) / avg, [t], t)
        states[("gap0",)] = (-0.8, [s0], s0)
        for k in range(1, n):
            nxt = {}
            for last, (sc, path, tl) in states.items():
                shift = path[0] - s0
                ek = exp[k] + shift * (1 - 0.6 * k / n)
                li = last if isinstance(last, int) else -1
                for j in range(li + 1, len(cand)):
                    t, s = cand[j]
                    if t < tl + 0.09:
                        continue
                    if t > ek + 1.3 * avg:
                        break
                    v = sc + 1.5 * s ** 0.7 - 1.2 * abs(t - ek) / avg
                    if j not in nxt or v > nxt[j][0]:
                        nxt[j] = (v, path + [t], t)
                te = max(ek, tl + 0.09)
                v = sc - 0.8 - 1.0 * abs(te - ek) / avg
                key = ("gap", li)
                if key not in nxt or v > nxt[key][0]:
                    nxt[key] = (v, path + [te], te)
            states = dict(sorted(nxt.items(), key=lambda kv: -kv[1][0])[:60])
        shift = 0.0
        sc, path, _ = max(states.values(), key=lambda q: q[0])
        path = [round(max(0.0, t - 1 / 60.0), 3) for t in path]   # 视觉领先 1 帧
        out[i] = [[w, float(t)] for w, t in zip(words, path)]
        rep.append((i, round(path[0] - s0, 3), [round(a - b, 2) for a, b in zip(path, exp)]))
    out.update({k: [list(x) for x in v] for k, v in MANUAL.items()})
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump({str(k): v for k, v in END.items()}, open(OUT.replace("lyric_words", "lyric_end"), "w"), indent=0)
    sh = np.array([r[1] for r in rep])
    dev = np.concatenate([np.abs(r[2]) for r in rep])
    print("lines %d, first-word shift median %+.3f s (|>0.1| %d), word deviation from even split: median %.3f s, p90 %.3f s"
          % (len(rep), np.median(sh), int((np.abs(sh) > 0.1).sum()), np.median(dev), np.percentile(dev, 90)))
    for i in (9, 12, 55, 56, 21, 24, 73):
        print(i, out[i])
    print("->", OUT)


if __name__ == "__main__":
    main()

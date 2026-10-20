# -*- coding: utf-8 -*-
"""
从"单一全局相位"改为"自适应拍轨 + 小节归组"

为什么改：
  beat_verify / phase_investigation 显示，全曲没有一个能同时对上底鼓和军鼓的
  单一相位。broadband 最优相位 0.084 s、kick 0.250 s，相差约半个八分音符；
  用各自相位的 35 ms 命中率分别只有 25.8% / 25.3%。
  说明 103 BPM 的速度可信，但"一个固定相位套全曲"的假设不成立——
  这在泽野弘之的编配里很常见（前奏、主歌、副歌的鼓组层次不同，
  还常有变速段落和过渡）。

做法：
  用 librosa 的动态规划拍轨得到逐拍时间（允许局部微调），
  再按"每 4 拍一小节"归组，用底鼓能量确定每小节第一拍，
  最后对齐到能量分段。输出可供 AE 直接读取的切点表。
"""
import os, sys, json
import numpy as np
import librosa

WAV = sys.argv[1] if len(sys.argv) > 1 else r"D:\备份\SawanoHiroyuki[nZk],Jean-Ken Johnny,TAKUMA - PROVANT.wav"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"D:\A\provant\data"
FPS = 24.0
SR, HOP, NFFT = 22050, 256, 2048

os.makedirs(OUT, exist_ok=True)
y, sr = librosa.load(WAV, sr=SR, mono=True)
dur = len(y) / SR

# ---------- 1. 自适应拍轨 ----------
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False, units='time')
tempo = float(np.atleast_1d(tempo)[0])

# ---------- 2. 各拍位的频带能量，用于找每小节第一拍 ----------
S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
fr = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
nfr = S.shape[1]

def band_energy(lo, hi):
    m = (fr >= lo) & (fr < hi)
    return S[m].sum(axis=0)

kick = band_energy(30, 120)
snare = band_energy(200, 600)
low_all = band_energy(30, 250)

# 在每拍附近 ±60 ms 取峰值，抗拍轨微小偏差
def at_beats(env, times, win=0.06):
    out = []
    for t in times:
        a = max(0, int((t - win) * SR / HOP))
        b = min(nfr, int((t + win) * SR / HOP) + 1)
        out.append(float(env[a:b].max()) if b > a else 0.0)
    return np.array(out)

n = len(beats)
k_b = at_beats(kick, beats)
s_b = at_beats(snare, beats)
l_b = at_beats(low_all, beats)

# ---------- 3. 每 4 拍一小节；用 4 个偏移里低频最强的一个当第一拍 ----------
best_off, best_score = 0, -1.0
for off in range(4):
    idx = np.arange(off, n, 4)
    if len(idx) < 8:
        continue
    # 第一拍应同时具备较强的底鼓与低频
    sc = float((k_b[idx] / (k_b.mean() + 1e-9)).mean() +
               (l_b[idx] / (l_b.mean() + 1e-9)).mean())
    if sc > best_score:
        best_score, best_off = sc, off

downbeats = beats[best_off::4]
bar_len_s = 4 * float(np.median(np.diff(beats)))

# 每个拍位的平均强度画像（用最优偏移作 0 号拍）
profile = {}
for name, arr in [("kick_30_120", k_b), ("snare_200_600", s_b), ("low_30_250", l_b)]:
    row = []
    for pos in range(4):
        idx = np.arange(best_off + pos, n, 4)
        idx = idx[idx < n]
        row.append(float((arr[idx] / (arr.mean() + 1e-9)).mean()) if len(idx) else 0.0)
    profile[name] = row

# ---------- 4. 能量分段（沿用研究包） ----------
segments = json.load(open(os.path.join(OUT, "..", "research", "PROVANT_research",
                                       "data", "music_map.json"), encoding="utf-8"))["segments"]

# 把每个分段边界吸附到最近的拍
def snap_to_beat(t):
    i = int(np.argmin(np.abs(beats - t)))
    return float(beats[i]), i

snapped = []
for s in segments:
    b0, i0 = snap_to_beat(s["start_s"])
    b1, i1 = snap_to_beat(s["end_s"])
    snapped.append({
        "id": s["id"], "label": s["label"],
        "raw_start_s": s["start_s"], "raw_end_s": s["end_s"],
        "snapped_start_s": b0, "snapped_end_s": b1,
        "start_beat_index": i0, "end_beat_index": i1,
        "start_shift_ms": round((b0 - s["start_s"]) * 1000, 1),
        "end_shift_ms": round((b1 - s["end_s"]) * 1000, 1),
        "start_is_downbeat": bool((i0 - best_off) % 4 == 0),
        "end_is_downbeat": bool((i1 - best_off) % 4 == 0),
        "bars": round((i1 - i0) / 4.0, 3),
    })

# ---------- 5. 切点表：以 2 小节为默认单位（研究包自相关显示 2 小节最强） ----------
cuts = []
for s in snapped:
    i0, i1 = s["start_beat_index"], s["end_beat_index"]
    if i0 >= len(beats):
        continue
    span_beats = i1 - i0
    if span_beats <= 4:
        step = 2          # 极短段：半小节
    elif span_beats <= 16:
        step = 8          # 2 小节
    else:
        step = 8          # 长段先按 2 小节，内部再分句
    idx = list(range(i0, i1, step))
    cuts.append({"segment": s["id"], "step_beats": step,
                 "times_s": [round(float(beats[i]), 4) for i in idx if i < len(beats)]})

res = {
    "source": os.path.basename(WAV),
    "duration_s": dur,
    "output_fps": FPS,
    "method": "librosa dynamic-programming beat tracking (adaptive per-beat phase), then 4-beat bar grouping",
    "why_not_single_phase": [
        "broadband best phase 0.084 s vs kick best phase 0.250 s — incompatible",
        "35 ms hit rate on real onsets only 25.8% (kick) / 25.3% (snare) with a fixed grid",
        "a fixed grid will therefore be audibly off on some sections",
    ],
    "tempo_bpm_from_tracker": tempo,
    "n_beats": n,
    "beat_span_s": [float(beats[0]), float(beats[-1])],
    "median_beat_s": float(np.median(np.diff(beats))),
    "bar_length_s": bar_len_s,
    "bar_length_at_24fps_frames": round(bar_len_s * FPS, 3),
    "downbeat_offset_within_beats": best_off,
    "downbeat_score": best_score,
    "bar_position_profile_relative": profile,
    "beat_times_s": [round(float(b), 4) for b in beats],
    "downbeat_times_s": [round(float(b), 4) for b in downbeats],
    "segments_snapped": snapped,
    "cut_table": cuts,
    "warnings": [
        "beat times are tracker output, NOT human-audition verified",
        "downbeat position within the bar is inferred from kick+low-band energy, not confirmed by ear",
        "4/4 remains a working hypothesis",
        "cut table gives grid-aligned candidates; every cut must still be checked by ear before locking",
    ],
}

path = os.path.join(OUT, "pulse_grid_adaptive.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# CSV for AE / spreadsheet
csv = os.path.join(OUT, "cut_table_adaptive.csv")
with open(csv, "w", encoding="utf-8-sig") as f:
    f.write("segment,bar_index,beat_index,time_s,frame_at_24fps,is_downbeat\n")
    for s in snapped:
        i0, i1 = s["start_beat_index"], s["end_beat_index"]
        for i in range(i0, min(i1, len(beats)), 4):
            t = float(beats[i])
            f.write("%s,%d,%d,%.4f,%d,%s\n" % (
                s["id"], (i - best_off) // 4, i, t, int(round(t * FPS)),
                "yes" if (i - best_off) % 4 == 0 else "no"))

print("tempo (tracker)  %.3f BPM" % tempo)
print("beats            %d, span %.3f - %.3f s, median %.4f s" % (n, beats[0], beats[-1], np.median(np.diff(beats))))
print("bar length       %.4f s = %.2f frames @24fps" % (bar_len_s, bar_len_s * FPS))
print("downbeat offset  %d beats into the tracked sequence (score %.3f)" % (best_off, best_score))
print()
print("== bar-position energy profile (mean 1.0) ==")
for k, v in profile.items():
    print("  %-14s %s" % (k, "  ".join("%.3f" % x for x in v)))
print()
print("== segments snapped to nearest tracked beat ==")
print("  %-4s %9s %9s %9s %9s  %7s %7s  %s" % (
    "id", "raw_start", "snap_start", "raw_end", "snap_end", "d_start", "d_end", "bars/downbeat"))
for s in snapped:
    print("  %-4s %9.3f %9.3f %9.3f %9.3f  %+6.0fms %+6.0fms  %5.2f  %s/%s" % (
        s["id"], s["raw_start_s"], s["snapped_start_s"], s["raw_end_s"], s["snapped_end_s"],
        s["start_shift_ms"], s["end_shift_ms"], s["bars"],
        "D" if s["start_is_downbeat"] else ".",
        "D" if s["end_is_downbeat"] else "."))
print()
print("wrote", path)
print("wrote", csv)

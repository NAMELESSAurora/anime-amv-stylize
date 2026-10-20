# -*- coding: utf-8 -*-
"""
PROVANT 拍点 / 相位 / 拍号 核验

补 PROVANT_MV_Research_v1 中标为未验证的 phase 与 time_signature。
原研究包用的是全带瞬态包络 + Fourier 周期；本脚本改为分频带瞬态包络，
因为底鼓/军鼓/踩镲落在不同的拍位上，只有拆开才能确定"哪一拍是强拍"。

可复跑：
  python beat_verify.py "<wav path>" <out dir>
"""
import sys, os, json
import numpy as np
import librosa

WAV_DEFAULT = r"D:\备份\SawanoHiroyuki[nZk],Jean-Ken Johnny,TAKUMA - PROVANT.wav"
OUT_DEFAULT = r"D:\A\provant\data"

SR, HOP, NFFT = 22050, 256, 2048

BANDS = {
    "kick_30_120":   (30, 120),
    "low_120_250":   (120, 250),
    "snare_200_600": (200, 600),
    "mid_600_2k":    (600, 2000),
    "high_3k_8k":    (3000, 8000),
}

TEMPOS = [68.90625, 82.03125, 102.992, 135.999, 206.71875]
WORKING_TEMPO = 102.992
WORKING_METER = 4


def build_envelopes(wav):
    y, sr = librosa.load(wav, sr=SR, mono=True)
    dur = len(y) / SR
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    fr = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    nfr = S.shape[1]

    env = {}
    for name, (lo, hi) in BANDS.items():
        m = (fr >= lo) & (fr < hi)
        b = np.log1p(S[m].sum(axis=0))
        env[name] = np.maximum(np.diff(b, prepend=b[0]), 0)

    b = np.log1p(S.sum(axis=0))
    env["broadband"] = np.maximum(np.diff(b, prepend=b[0]), 0)

    # 归一化到均值 1，这样不同速度的 grid/mean 可以直接比
    for k in env:
        env[k] = env[k] / (env[k].mean() + 1e-12)
    return env, dur, nfr


def comb(env_band, dur, nfr, tempo, phase_scan=200):
    """在给定速度下扫相位，返回格点上的平均瞬态强度。"""
    beat = 60.0 / tempo
    best = (-1.0, None)
    for ph in np.linspace(0.0, beat, phase_scan, endpoint=False):
        times = ph + beat * np.arange(0, int((dur - ph) / beat))
        idx = np.clip((times * SR / HOP).astype(int), 0, nfr - 1)
        s = float(env_band[idx].mean())
        if s > best[0]:
            best = (s, float(ph))
    return {"tempo_bpm": tempo, "beat_s": beat,
            "grid_mean_over_envelope_mean": best[0], "phase_s": best[1]}


def main():
    wav = sys.argv[1] if len(sys.argv) > 1 else WAV_DEFAULT
    out = sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT
    os.makedirs(out, exist_ok=True)

    env, dur, nfr = build_envelopes(wav)
    res = {
        "source": os.path.basename(wav),
        "duration_s": dur,
        "analysis_sample_rate_hz": SR,
        "hop_samples": HOP,
        "hop_s": HOP / SR,
        "n_fft": NFFT,
        "bands_hz": {k: list(v) for k, v in BANDS.items()},
        "method": "band-limited spectral-flux onset envelopes, each normalised to mean=1",
        "not_verified": [
            "human audition of the chosen phase",
            "whether every grid point lands on a real drum hit",
        ],
    }

    res["tempo_scan"] = {
        name: [comb(env[name], dur, nfr, t) for t in TEMPOS] for name in env
    }

    # 以底鼓相位为锚，给出每小节 4 个拍位在各频带上的强度画像
    kick_ph = comb(env["kick_30_120"], dur, nfr, WORKING_TEMPO)["phase_s"]
    beat = 60.0 / WORKING_TEMPO
    bar = WORKING_METER * beat

    profile = {}
    for name in env:
        row = []
        for pos in range(WORKING_METER):
            times = kick_ph + beat * (
                np.arange(0, int((dur - kick_ph) / beat)) + pos / float(WORKING_METER)
            )
            times = times[times < dur]
            idx = np.clip((times * SR / HOP).astype(int), 0, nfr - 1)
            row.append(float(env[name][idx].mean()))
        profile[name] = row

    res["bar_position_profile"] = {
        "tempo_bpm": WORKING_TEMPO,
        "meter_assumed": "4/4 (working hypothesis)",
        "beat_s": beat,
        "bar_s": bar,
        "anchor_phase_s": kick_ph,
        "meaning": "positions 0..3 = the four beats of a bar, 0 = downbeat candidate",
        "values": profile,
    }

    res["downbeat_contrast"] = {
        k: {
            "pos0": v[0],
            "mean_pos1_3": float(np.mean(v[1:])),
            "ratio_pos0_over_others": v[0] / (float(np.mean(v[1:])) + 1e-12),
        }
        for k, v in profile.items()
    }

    # 半拍（八分）脉冲检查：Sawano 的曲子里常有驱动性八分音型
    eighth = {}
    for name in ["kick_30_120", "snare_200_600", "broadband"]:
        ph = comb(env[name], dur, nfr, WORKING_TEMPO)["phase_s"]
        times = ph + (beat / 2.0) * np.arange(0, int((dur - ph) / (beat / 2.0)))
        idx = np.clip((times * SR / HOP).astype(int), 0, nfr - 1)
        eighth[name] = {
            "on_quarter_grid": comb(env[name], dur, nfr, WORKING_TEMPO)["grid_mean_over_envelope_mean"],
            "on_eighth_grid": float(env[name][idx].mean()),
        }
    res["eighth_note_check"] = eighth

    # 稳健性：在最优相位附近多大的偏移仍然保持 97% 强度
    rob = {}
    for name in ["kick_30_120", "snare_200_600", "broadband"]:
        e = env[name]
        scored = []
        for ph in np.arange(0.0, beat, 0.001):
            times = ph + beat * np.arange(0, int((dur - ph) / beat))
            idx = np.clip((times * SR / HOP).astype(int), 0, nfr - 1)
            scored.append((float(e[idx].mean()), float(ph)))
        scored.sort(reverse=True)
        top = scored[0][0]
        good = [p for s, p in scored if s > top * 0.97]
        rob[name] = {
            "best_phase_s": scored[0][1],
            "best_strength": top,
            "plateau_width_ms": (max(good) - min(good)) * 1000.0,
            "plateau_range_s": [min(good), max(good)],
        }
    res["phase_robustness"] = rob

    res["per_bar_rms_dbfs"] = None  # 见 per_bar_map.py

    path = os.path.join(out, "beat_verify.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ---- 控制台摘要 ----
    print("== tempo scan (broadband): grid/mean, higher = stronger periodicity ==")
    for r in res["tempo_scan"]["broadband"]:
        print("  %9.3f bpm   %.4f   phase %.4f s"
              % (r["tempo_bpm"], r["grid_mean_over_envelope_mean"], r["phase_s"]))
    print()
    print("== bar-position profile at %.3f bpm (pos0 = downbeat candidate) ==" % WORKING_TEMPO)
    for k, v in profile.items():
        print("  %-14s %s" % (k, "  ".join("%.3f" % x for x in v)))
    print()
    print("== downbeat contrast (pos0 / mean(pos1..3)) ==")
    for k, v in res["downbeat_contrast"].items():
        print("  %-14s %.3f" % (k, v["ratio_pos0_over_others"]))
    print()
    print("== eighth-note check ==")
    for k, v in eighth.items():
        print("  %-14s quarter %.3f  eighth %.3f  ratio %.3f"
              % (k, v["on_quarter_grid"], v["on_eighth_grid"],
                 v["on_eighth_grid"] / (v["on_quarter_grid"] + 1e-12)))
    print()
    print("== phase robustness ==")
    for k, v in rob.items():
        print("  %-14s best %.4f s  plateau %.0f ms"
              % (k, v["best_phase_s"], v["plateau_width_ms"]))
    print()
    print("wrote", path)


if __name__ == "__main__":
    main()

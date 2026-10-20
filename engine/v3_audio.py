# -*- coding: utf-8 -*-
"""
v3 原声层（用户 10-19 第 8 条："台词可以多加一些女性的台词，并且现有的台词插入很浑浊而且不用刻意压低音乐，并且台词声音可以适当大一点"）。

和 v2 的区别：
  · 台词换成 EDL 里 cfg["dialog"] 的女性句子（dialog_harvest：MDX 人声分离 → Silero VAD → 基频 >185 Hz → Whisper 转写，人挑），
    取分离后的干声（E:\\provant_work\\audio\\sep\\<yid>.wav），不再带源里的配乐 / 音效 → 不浑
  · 音乐一律不压（不 duck）；台词按所在位置音乐的响度定电平：比音乐 RMS 高 6 dB，限在 -20…-10 dBFS（RMS，按有声部分算）
  · 台词处理：高通 90 Hz、300 Hz 处 -2.5 dB（去闷）、3.2 kHz 处 +3.5 dB（咬字）、拉到目标电平后 4:1 压缩（3 ms 起）+ 台词总线 -5 dBFS 限制、20 ms 淡入 / 120 ms 淡出
  · 爆点音效（hit / rise / accent）保留、也不 duck，只放在正片源上（同步取）
  · 峰值限制 -1 dBFS
输出：out/v3/mix_v3.wav、data/audio_events_v3.json；--mux 合成 out/PROVANT_AMV_v3.mp4
"""
import glob, json, math, os, subprocess, sys
import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(__file__))
import v2_audio as V2

ROOT = V2.ROOT
FF = V2.FF
SR = V2.SR
KEY = next((a for a in sys.argv[1:] if a in ("v5", "v6", "v7", "v8", "v9", "v10")), "v5")   # EDL 版本：v5 → 成片 v3，v6 → v4，v7 → v5，v8 → v6
VER = {"v5": "v3", "v6": "v4", "v7": "v5", "v8": "v6", "v9": "v7", "v10": "v8"}[KEY]
EDL = json.load(open(os.path.join(ROOT, "data", "edl_%s.json" % KEY), encoding="utf-8"))
V2.EDL = EDL                                               # v2 的事件挑选函数读的是模块级 EDL
SEP = r"E:\provant_work\audio\sep"
OUT = os.path.join(ROOT, "out", VER)


# ---------------------------------------------------------------- RBJ 双二阶滤波
def biquad(kind, f0, q=0.707, gain_db=0.0, sr=SR):
    A = 10 ** (gain_db / 40.0); w = 2 * math.pi * f0 / sr; al = math.sin(w) / (2 * q); cw = math.cos(w)
    if kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + al, -2 * cw, 1 - al]
    else:                                                   # peak
        b = [1 + al * A, -2 * cw, 1 - al * A]; a = [1 + al / A, -2 * cw, 1 - al / A]
    return np.array(b) / a[0], np.array(a) / a[0]


def eq_voice(x):
    for kind, f0, q, g in (("hp", 90, 0.707, 0), ("hp", 90, 0.707, 0), ("pk", 300, 1.0, -2.5), ("pk", 3200, 0.9, 3.5)):
        b, a = biquad("hp" if kind == "hp" else "pk", f0, q, g)
        x = lfilter(b, a, x, axis=0)
    from scipy.signal import butter, sosfilt                # 11 kHz 低通：磨掉人声分离留下的高频"水声"
    x = sosfilt(butter(4, 11000, btype="lowpass", fs=SR, output="sos"), x, axis=0)
    return x.astype(np.float32)


def compress(x, thr_db=-24.0, ratio=3.0, att_s=0.010, rel_s=0.120):
    """包络检测（默认 10 ms 起 / 120 ms 放）的压缩。"""
    m = np.sqrt((x ** 2).mean(1))
    att, rel = math.exp(-1 / (att_s * SR)), math.exp(-1 / (rel_s * SR))
    env = np.empty_like(m); y = 0.0
    for i in range(0, len(m), 48):
        v = float(m[i:i + 48].max()); c = att if v > y else rel
        y = v + (y - v) * (c ** 48); env[i:i + 48] = y
    lv = 20 * np.log10(env + 1e-9)
    gr = np.where(lv > thr_db, (lv - thr_db) * (1 - 1 / ratio), 0.0)
    return (x * (10 ** (-gr / 20.0))[:, None]).astype(np.float32)


def limit(x, ceil_db, look_s=0.005, rel_s=0.06):
    """前瞻峰值限制：增益只在峰前 look_s 内压下去，按 rel_s 放回。返回 (输出, 增益曲线)。"""
    n = len(x)
    pk = np.abs(x).max(1)
    need = np.minimum(1.0, 10 ** (ceil_db / 20.0) / np.maximum(pk, 1e-6))
    look = int(look_s * SR)
    g = np.empty(n, np.float32); y = 1.0; rr = math.exp(-1 / (rel_s * SR))
    for i in range(0, n, 48):
        x_ = float(need[max(0, i - look):i + 48 + look].min())
        y = x_ if x_ < y else x_ + (y - x_) * (rr ** 48)
        g[i:i + 48] = y
    return x * g[:, None], g


def carve(music, dbus, depth_db=6.0, lo=250.0, hi=4500.0):
    """台词清晰度（v4，用户："台词还是很浑浊"）：只在台词响的时候，把音乐中置声道（M）的 250–4500 Hz 挖低 depth_db；
    S 声道、低频、高频都不动 —— 听感上音乐没有被压低，只是给人声让出中频位置。零相位分频，M_band + M_rest 严格等于原 M。"""
    from scipy.signal import butter, sosfiltfilt
    m = (music[:, 0] + music[:, 1]) * 0.5; sd = (music[:, 0] - music[:, 1]) * 0.5
    sos = butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    band = sosfiltfilt(sos, m).astype(np.float32)
    rest = m - band
    v = np.sqrt((dbus ** 2).mean(1))
    att, rel = math.exp(-1 / (0.040 * SR)), math.exp(-1 / (0.300 * SR))
    env = np.empty_like(v); y = 0.0
    for i in range(0, len(v), 48):
        x = 1.0 if v[i:i + 48].max() > 0.01 else 0.0
        c = att if x > y else rel
        y = x + (y - x) * (c ** 48); env[i:i + 48] = y
    g = 10 ** (-depth_db * env / 20.0)
    m2 = rest + band * g
    return np.stack([m2 + sd, m2 - sd], 1).astype(np.float32)


def stem(yid, s0, s1):
    p = os.path.join(SEP, yid + ".wav")
    r = subprocess.run([FF, "-v", "error", "-ss", "%.4f" % s0, "-t", "%.4f" % (s1 - s0), "-i", p, "-ac", "2", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.float32)
    return a[:a.size // 2 * 2].reshape(-1, 2).copy()


def active_rms_db(x):
    m = np.sqrt((x ** 2).mean(1)); hop = int(0.02 * SR)
    e = np.array([np.sqrt((m[i:i + hop] ** 2).mean()) for i in range(0, len(m) - hop, hop)])
    on = e[e > e.max() * 0.15]
    return 20 * np.log10(np.sqrt((on ** 2).mean()) + 1e-9) if on.size else -60.0


def load_music():
    wav = [q for q in glob.glob(r"D:\备份\*.wav") if "PROVANT" in os.path.basename(q)][0]
    r = subprocess.run([FF, "-v", "error", "-i", wav, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    music = np.frombuffer(r.stdout, np.float32).reshape(-1, 2).copy()
    n = int(round(EDL["T1"] * SR))
    return np.pad(music, ((0, max(0, n - len(music))), (0, 0)))[:n]


def main():
    music = load_music() * np.float32(10 ** (-2.0 / 20))         # 母带峰值到 0 dBFS、重采样后还有 +0.9 dB 的样间过冲：先让 2 dB，限制器只抓叠加的瞬态
    n = len(music)
    fx = np.zeros_like(music)
    ev = [dict(e, duck=0.0, gain=e["gain"] - 4.0) for e in V2.events() if e["kind"] in ("hit", "rise", "accent")]
    for e in ev:
        a = V2.audio(e["yid"], e["src"], e["dur"])
        if a is None:
            continue
        L = len(a); fi = int(e.get("fade_in", 0.02) * SR); fo = int(min(0.25, e["dur"] / 3) * SR)
        env = np.ones(L, np.float32)
        if fi:
            env[:fi] = np.linspace(0, 1, fi)
        env[L - fo:] = np.linspace(1, 0, fo)
        i0 = int(e["t"] * SR); i1 = min(n, i0 + L)
        mus = 20 * np.log10(np.sqrt((music[i0:i1] ** 2).mean()) + 1e-9)
        if mus > -11.0:                                     # 音乐本身已经顶满（高潮爆点）：音效只做质感，不然限制器会把爆点压下去
            e["gain"] = min(e["gain"], -18.0)
        fx[i0:i1] += (a * env[:, None] * (10 ** (e["gain"] / 20.0)))[:i1 - i0]
    dbus = np.zeros_like(music)                            # 台词总线：自己先限到 -5 dBFS（语音峰值因子大，别让总限制器替它压音乐）
    for d in EDL.get("dialog", []):
        x = eq_voice(stem(d["yid"], d["s0"], d["s1"]))
        L = len(x); fi, fo = int(0.02 * SR), int(0.12 * SR)
        env = np.ones(L, np.float32); env[:fi] = np.linspace(0, 1, fi); env[L - fo:] = np.linspace(1, 0, fo)
        x = x * env[:, None]
        i0 = int(d["t"] * SR); i1 = min(n, i0 + L)
        mus = 20 * np.log10(np.sqrt((music[i0:i1] ** 2).mean()) + 1e-9)
        tgt = float(np.clip(mus + 7.0, -15.0, -9.0))
        g = tgt - active_rms_db(x)                          # 先拉到目标电平，再在目标附近压（4:1、3 ms 起），压掉语音 15 dB 的峰值因子
        x = compress(x * 10 ** (g / 20.0), thr_db=tgt - 3.0, ratio=4.0, att_s=0.003, rel_s=0.10)
        mk = tgt - active_rms_db(x)
        x = x * 10 ** (mk / 20.0); g += mk
        dbus[i0:i1] += x[:i1 - i0]
        ev.append(dict(kind="dialog", t=d["t"], yid=d["yid"], src=d["s0"], dur=round(d["s1"] - d["s0"], 3), gain=round(float(g), 1), duck=0.0,
                       music_db=round(float(mus), 1), target_db=round(tgt, 1)))
    dbus, gd = limit(dbus, -5.0, look_s=0.003, rel_s=0.05)
    music = carve(music, dbus)
    out, g = limit(music + fx + dbus, -1.0)                # 总线峰值限制 -1 dBFS
    on = np.abs(dbus).max(1) > 1e-4
    print("dialog limiter: max GR %.1f dB, mean GR while talking %.1f dB" % (-20 * math.log10(gd.min()), -20 * math.log10(gd[on].mean())))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "mix_%s.wav" % VER)
    subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_s24le", p],
                   input=out.astype(np.float32).tobytes(), check=True)
    ev.sort(key=lambda e: e["t"])
    json.dump(ev, open(os.path.join(ROOT, "data", "audio_events_%s.json" % VER), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for e in ev:
        print("%-7s %7.3f  %-12s @%7.2f  %.2fs  %+5.1fdB%s" % (e["kind"], e["t"], e["yid"], e["src"], e["dur"], e["gain"],
                                                         ("  music %.1f → voice %.1f dBFS" % (e["music_db"], e["target_db"])) if "music_db" in e else ""))
    print("mix ->", p, "peak %.3f, limiter min gain %.2f dB @ %.2f s, >1 dB GR: %.2f s total" % (np.abs(out).max(), 20 * math.log10(max(1e-6, g.min())), g.argmin() / SR, (g < 10 ** (-1 / 20)).sum() / SR))
    return p


def mux(mixp):
    v = os.path.join(OUT, EDL["out"])
    o = os.path.join(ROOT, "out", "PROVANT_AMV_%s.mp4" % VER)
    subprocess.run([FF, "-v", "error", "-y", "-i", v, "-i", mixp, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                    "-shortest", o], check=True)
    print("->", o)


if __name__ == "__main__":
    mp = main()
    if "--mux" in sys.argv:
        mux(mp)

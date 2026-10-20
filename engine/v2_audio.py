# -*- coding: utf-8 -*-
"""
原声层（用户 10-18："成片可以适当加一点原声"）。

只取"正片 / 战斗剪辑"的原声（HF 剧场版、FSF Whispers of Dawn / 战斗、FGO 与 Apocrypha 战斗片段）——
OP / ED / PV / 纪念影像的音轨本身就是别的歌，叠上去会和 PROVANT 打架，一律不用。全部按画面同步取（src_t 映射，只取 1.0× 速度的镜头）。
  hit      6 个爆点（PUNCH）：该镜头源里最强的瞬态起 ~1 s（爆炸 / 刀剑 / 嘶吼），-3 dB，音乐压 4 dB
  rise     换气定格（127.57–128.128）：爆点镜头源里爆点前那一段（蓄力声），淡入
  accent   高潮段 burst / 冲击帧镜头的开头瞬态，0.5 s，-7 dB（最多 8 处）
  line     音乐器乐空档（30.98–38.24、100.89–108.23）+ 片尾（165.6–171）：Silero VAD 找画面上那个镜头里的真对白（≥0.6 s），
           同步放，-1 dB，音乐压 9 dB
混音：numpy，48 kHz 立体声；音乐按原声包络自动压低（起 10 ms / 放 250 ms）；峰值限制到 -1 dBFS。
输出：out/v2/mix_v2.wav、data/audio_events.json（每条：类型 / 时间 / 镜头 / 源 / 源时间 / 增益），再和视频合成成片。
用法: python scripts/v2_audio.py [--mux]
"""
import glob, json, math, os, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
ROOT = r"D:\A\provant"
FF = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
SR = 48000
FPS = 60
_E4 = os.path.join(ROOT, "data", "edl_v4.json")          # 单独跑 v2 时用；v3_audio 会把当前 EDL 塞进 V2.EDL（edl_v4 已归档）
EDL = json.load(open(_E4, encoding="utf-8")) if os.path.exists(_E4) else None
SRC = json.load(open(os.path.join(ROOT, "data", "audio_sources.json"), encoding="utf-8"))
FILM = {  # 有正片原声（对白 / 音效）的源
    "tlKnVXKaZYs", "jqQkurxOj-Q", "vUqzHyadZAA", "vSaILO_7UVk", "a8QQN9RDrSk", "B5DdMAhWCsE", "aT7kG7zdkak", "9c4ozWgw3Ns",
    "3U3DFMZBddA", "rZOxWc1nGt8", "oZlRdc9gxE4", "cpxwZP1GfAs", "hQcHTLb-FJs", "tY4SZ_NKKb4",            # HF 剧场版（用户的 hf 文件夹）
    "EpLKseoUV6Q", "DlyEbc5AMUw",                                                                         # FSF
    "_2j1V81wuR0", "deONLS7FcVI", "Yuq57lj3LMU", "qiANLuvPb9o", "o1H-NNBjHbE", "Cf9uMIffDTQ",            # FGO 战斗 / 终局
    "BWiA7y_XNHY", "kgqkkGuyRjU", "RxmNey6OU_s",                                                          # Apocrypha 战斗
}
GAPS = [(30.98, 38.24), (100.89, 108.23), (165.6, 171.0)]


def fr(t):
    return int(round(t * FPS))


def src_t(s, t):
    rel = t - s["a"]
    if s.get("hold"):
        rel = math.floor(rel * s["hold"]) / s["hold"]
    return float(np.clip(s["src_a"] + rel * s["speed"], s["lo"], s["hi"]))


def shot_at(t):
    return next((s for s in EDL["shots"] if s["a"] <= t < s["b"]), None)


def audio(yid, t0, dur, sr=SR, ch=2):
    p = SRC.get(yid)
    if not p or dur <= 0:
        return None
    r = subprocess.run([FF, "-v", "error", "-ss", "%.4f" % max(0, t0), "-t", "%.4f" % dur, "-i", p, "-vn", "-ac", str(ch), "-ar", str(sr),
                        "-f", "f32le", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.float32)
    if a.size < ch * 64:
        return None
    return a[:a.size // ch * ch].reshape(-1, ch).copy()


# ---------------------------------------------------------------- VAD
_VAD = None


def vad(mono16):
    """Silero VAD v5（onnx）：每 512 样本（32 ms）一个语音概率。"""
    global _VAD
    import onnxruntime as ort
    if _VAD is None:
        _VAD = ort.InferenceSession(os.path.join(ROOT, "models", "silero-vad", "onnx", "model.onnx"), providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), np.float32); ctx = np.zeros(64, np.float32); out = []
    for k in range(0, len(mono16) - 512 + 1, 512):
        x = np.concatenate([ctx, mono16[k:k + 512]])[None].astype(np.float32)
        p, state = _VAD.run(None, {"input": x, "state": state, "sr": np.array(16000, np.int64)})
        ctx = mono16[k + 512 - 64:k + 512]
        out.append(float(p[0, 0]))
    return np.array(out)


def transient(a):
    """最强瞬态的位置（秒）：短时能量的正向差分最大处。"""
    m = np.abs(a).mean(1)
    hop = 480
    e = np.array([m[i:i + hop].mean() for i in range(0, len(m) - hop, hop)])
    if len(e) < 3:
        return 0.0
    d = np.diff(e)
    return float(np.argmax(d) * hop / SR)


# ---------------------------------------------------------------- 选事件
def events():
    ev = []
    for t in EDL.get("punch", []):                         # hit
        s = shot_at(t + 0.01)
        if s and s["yid"] in FILM and s.get("speed", 1) >= 0.9:
            st = src_t(s, t)
            a = audio(s["yid"], st - 0.3, 1.6)
            if a is not None:
                off = transient(a[:int(0.9 * SR)]) - 0.3          # 瞬态对齐到爆点
                ev.append(dict(kind="hit", t=round(t, 3), shot=s["id"], yid=s["yid"], src=round(st + off, 3), dur=1.1, gain=-3.0, duck=4.0))
    for g in [s for s in EDL["shots"] if s.get("bars")]:  # rise：每个换气段（下一镜是正片才有）
        d1 = shot_at(g["b"] + 0.01)
        if d1 and d1["yid"] in FILM:
            st = src_t(d1, d1["a"])
            ev.append(dict(kind="rise", t=round(g["a"], 3), shot=d1["id"], yid=d1["yid"], src=round(max(0.0, st - (g["b"] - g["a"])), 3),
                           dur=round(g["b"] - g["a"], 3), gain=-6.0, duck=3.0, fade_in=0.35))
    n_acc = 0                                              # accent：高潮段冲击镜头开头
    for s in EDL["shots"]:
        if n_acc >= 8:
            break
        if s.get("hot") and (s.get("burst") or s.get("impact")) and s["yid"] in FILM and s.get("speed", 1) >= 0.9:
            if any(abs(s["a"] - e["t"]) < 0.3 for e in ev):
                continue
            st = src_t(s, s["a"])
            a = audio(s["yid"], st, 0.6)
            if a is not None and np.abs(a).max() > 0.05:
                ev.append(dict(kind="accent", t=round(s["a"], 3), shot=s["id"], yid=s["yid"], src=round(st, 3), dur=0.5, gain=-7.0, duck=2.5))
                n_acc += 1
    for g0, g1 in GAPS:                                    # line：空档里的真对白
        best = None
        for s in EDL["shots"]:
            a0, a1 = max(g0, s["a"]), min(g1, s["b"])
            if a1 - a0 < 0.8 or s["yid"] not in FILM or abs(s.get("speed", 1) - 1) > 0.02 or s.get("hold"):
                continue
            st0 = src_t(s, a0)
            mono = audio(s["yid"], st0, a1 - a0, sr=16000, ch=1)
            if mono is None:
                continue
            pr = vad(mono[:, 0])
            print("   vad %-9s %-5s %6.2f–%6.2f  max %.2f  >0.5: %.1fs" % (s["id"], s["code"], a0, a1, pr.max() if len(pr) else 0,
                                                                     (pr > 0.5).sum() * 0.032))
            on = pr > 0.5
            k = 0
            while k < len(on):                              # 连续语音段
                if on[k]:
                    j = k
                    while j < len(on) and on[j]:
                        j += 1
                    L = (j - k) * 0.032
                    if L >= 0.5:
                        sc = L * float(pr[k:j].mean())
                        if best is None or sc > best[0]:
                            best = (sc, dict(kind="line", t=round(a0 + k * 0.032 - 0.05, 3), shot=s["id"], yid=s["yid"],
                                             src=round(st0 + k * 0.032 - 0.05, 3), dur=round(min(L + 0.25, a1 - a0 - k * 0.032), 3),
                                             gain=-1.0, duck=9.0, vad=round(float(pr[k:j].mean()), 2)))
                    k = j
                k += 1
        if best:
            ev.append(best[1])
    # voice：画面上没有可同步的对白时，从言峰独白片段（cpxwZP1GfAs；画面有烧录字幕不用，音轨是干净独白）取两句，
    # 作为旁白放进两段器乐空档（VAD 概率高、前后有停顿的 1.5–3.2 s 段）
    if not any(e["kind"] == "line" for e in ev):
        mono = audio("cpxwZP1GfAs", 0.0, 109.0, sr=16000, ch=1)
        if mono is not None:
            pr = vad(mono[:, 0]); segs = []
            k = 0
            while k < len(pr):
                if pr[k] > 0.7:
                    j = k
                    while j < len(pr) and pr[j] > 0.5:
                        j += 1
                    L = (j - k) * 0.032
                    pre = pr[max(0, k - 10):k].mean() if k > 0 else 0; post = pr[j:j + 10].mean() if j < len(pr) else 0
                    if 1.5 <= L <= 3.2 and pre < 0.3 and post < 0.3:
                        segs.append((float(pr[k:j].mean()), k * 0.032, L))
                    k = j
                k += 1
            segs.sort(reverse=True)
            picked = []
            for sc, t0, L in segs:
                if all(abs(t0 - q[1]) > 8 for q in picked):
                    picked.append((sc, t0, L))
                if len(picked) == 2:
                    break
            picked.sort(key=lambda q: q[1])
            for (sc, t0, L), at in zip(picked, (32.4, 102.2)):
                ev.append(dict(kind="voice", t=at, shot="-", yid="cpxwZP1GfAs", src=round(t0 - 0.08, 3), dur=round(L + 0.25, 3), gain=0.0, duck=8.0,
                               vad=round(sc, 2)))
    return sorted(ev, key=lambda e: e["t"])


# ---------------------------------------------------------------- 混音
def mix(ev):
    wav = [q for q in glob.glob(r"D:\备份\*.wav") if "PROVANT" in os.path.basename(q)][0]
    r = subprocess.run([FF, "-v", "error", "-i", wav, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    music = np.frombuffer(r.stdout, np.float32).reshape(-1, 2).copy()
    n = int(round(EDL["T1"] * SR))
    music = np.pad(music, ((0, max(0, n - len(music))), (0, 0)))[:n]
    fx = np.zeros_like(music)
    duck = np.zeros(n, np.float32)
    for e in ev:
        a = audio(e["yid"], e["src"], e["dur"])
        if a is None:
            continue
        L = len(a)
        fi = int(e.get("fade_in", 0.02) * SR); fo = int(min(0.25, e["dur"] / 3) * SR)
        env = np.ones(L, np.float32)
        env[:fi] = np.linspace(0, 1, fi) if fi else 1
        env[L - fo:] = np.linspace(1, 0, fo)
        a = a * env[:, None] * (10 ** (e["gain"] / 20.0))
        i0 = int(e["t"] * SR)
        i1 = min(n, i0 + L)
        fx[i0:i1] += a[:i1 - i0]
        d = np.zeros(n, np.float32); d[i0:i1] = e["duck"] * env[:i1 - i0]
        duck = np.maximum(duck, d)
    # 压音乐：起 10 ms / 放 250 ms 的平滑
    att, rel = math.exp(-1 / (0.010 * SR)), math.exp(-1 / (0.250 * SR))
    sm = np.empty_like(duck); y = 0.0
    for i in range(0, n, 48):                              # 1 ms 步长近似
        x = float(duck[i:i + 48].max())
        c = att if x > y else rel
        y = x + (y - x) * (c ** 48)
        sm[i:i + 48] = y
    out = music * (10 ** (-sm / 20.0))[:, None] + fx
    pk = np.abs(out).max(1)                                # 峰值限制 -1 dBFS（5 ms 预看，100 ms 释放）
    lim = 10 ** (-1 / 20.0)
    look = int(0.005 * SR)
    need = np.minimum(1.0, lim / np.maximum(pk, 1e-6))
    need = np.minimum.accumulate(np.concatenate([need[look:], np.ones(look)])[::-1])[::-1] if False else need
    g = np.empty(n, np.float32); y = 1.0; rr = math.exp(-1 / (0.1 * SR))
    for i in range(0, n, 48):
        x = float(need[max(0, i - look):i + 48 + look].min())
        y = x if x < y else x + (y - x) * (rr ** 48)
        g[i:i + 48] = y
    out = out * g[:, None]
    p = os.path.join(ROOT, "out", "v2", "mix_v2.wav")
    subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_s24le", p],
                   input=out.astype(np.float32).tobytes(), check=True)
    print("mix ->", p, "peak %.3f" % np.abs(out).max(), "duck max %.1f dB" % sm.max())
    return p


def mux(mixp):
    v = os.path.join(ROOT, "out", "v2", EDL["out"])
    o = os.path.join(ROOT, "out", "PROVANT_AMV_v2.mp4")
    subprocess.run([FF, "-v", "error", "-y", "-i", v, "-i", mixp, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                    "-shortest", o], check=True)
    print("->", o)


if __name__ == "__main__":
    ev = events()
    json.dump(ev, open(os.path.join(ROOT, "data", "audio_events.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for e in ev:
        print("%-6s %7.3f  %-9s %s @%.2f  %.2fs  %+.0fdB  duck %.0f%s" % (e["kind"], e["t"], e["shot"], e["yid"], e["src"], e["dur"], e["gain"],
                                                                      e["duck"], ("  vad %.2f" % e["vad"]) if "vad" in e else ""))
    mp = mix(ev)
    if "--mux" in sys.argv:
        mux(mp)

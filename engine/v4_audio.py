# -*- coding: utf-8 -*-
"""
v9 成片混音（edl_v11 / KEY v11 → out/v9）。用户 10-20：
  "可以加上一些原片的环境音" / "intro 可以不放歌先做点简短介绍然后再引入歌名和 title 再开始放歌" /
  "中间可以有一些 break 音乐淡出…再引入音乐，整个过程要自然流畅" / "尾声淡出可以留长一点时间，音乐结束后还可以留一点画面以及环境音"
结构：
  · 歌按 EDL["song_map"] 分段放（[成片起点, 歌起点, 歌终点]），中间的 break 处歌自己衰减到底再切开
  · native_audio：前奏召唤咏唱、间奏片段、尾声 —— 原片整轨（对白 + 配乐 + 音效）按有声 RMS 归一到目标电平
  · 环境底：歌段里每个正片源、接近原速（±12%）的镜头，同步取原声，高通 150 Hz、归一到 -31 dBFS（歌很安静处 -27），40 ms 交叉；
    台词 / native 段 / 爆点音效附近不放
  · 爆点音效（hit / rise / accent）、台词总线（EQ + 压缩 + -5 dBFS 限制）、音乐中频挖槽（carve）、总线 -1 dBFS 限制 —— 沿用 v3_audio
  · v10 第二版（用户看过 v10："素材原声不用单独提取环境音和人声（这样会导致失真），将原声的背景音压低一点，羽蛇神那一段…按原声放出"；
    "现有的原声特别是角色的语音有些失真"）：魁札尔 break = 原样（只做静态增益、峰值封顶）；其余原声 / 环境底 / 爆点 / 台词 = 原声、背景（配乐）压低
    6 dB（压在歌上的台词 9 dB；sep_bandit.bg）；压缩器 / 限制器增益逐样本平滑、压缩改 2:1 慢起；电平按峰值封顶，不靠限制器硬压
  · 环境底去重（用户："冲田总司的片段语音存在重复 0:00:43–0:00:47"）：同一源的环境底按源时间只读一次（连着的慢放镜头接着上一段往下读，
    不再各自从镜头源起点重读），被台词用掉的源区间在环境底里静音
"""
import glob, json, math, os, subprocess, sys
import numpy as np
from scipy.signal import lfilter, butter, sosfilt

sys.path.insert(0, os.path.dirname(__file__))
import v2_audio as V2
import sep_bandit as SB                                      # 去配乐（用户 10-20："仅保留人物音和环境音"）

ROOT = V2.ROOT
FF = V2.FF
SR = V2.SR
KEY = next((a for a in sys.argv[1:] if a.startswith("v1")), "v11")
VER = {"v11": "v9", "v12": "v10"}[KEY]
EDL = json.load(open(os.path.join(ROOT, "data", "edl_%s.json" % KEY), encoding="utf-8"))
V2.EDL = EDL
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


def eq_voice(x, sep=False):
    """台词 EQ（v10：原声不再分离，提升收小）。sep=True（MDX 分离干声，v7 的做法）：加回 11 kHz 低通磨掉分离残留的高频"水声"。"""
    for kind, f0, q, g in (("hp", 90, 0.707, 0), ("pk", 300, 1.0, -1.5), ("pk", 3200, 0.9, 2.0)):
        b, a = biquad("hp" if kind == "hp" else "pk", f0, q, g)
        x = lfilter(b, a, x, axis=0)
    if sep:
        x = sosfilt(butter(4, 11000, btype="lowpass", fs=SR, output="sos"), x, axis=0)
    return x.astype(np.float32)


def _smooth_gain(gb, n):
    """块增益（每 48 样本一个，取块中心）→ 逐样本线性插值。"""
    xs = np.arange(len(gb)) * 48 + 24
    return np.interp(np.arange(n), xs, gb).astype(np.float32)


def compress(x, thr_db=-24.0, ratio=2.0, att_s=0.010, rel_s=0.150):
    """包络检测压缩；增益逐样本平滑（v10 用户："原声特别是角色的语音有些失真"——原来 1 ms 一跳的阶梯增益 + 13 dB 压缩是元凶之一）。"""
    m = np.sqrt((x ** 2).mean(1))
    att, rel = math.exp(-1 / (att_s * SR)), math.exp(-1 / (rel_s * SR))
    nb = (len(m) + 47) // 48
    env = np.empty(nb, np.float64); y = 0.0
    for k in range(nb):
        v = float(m[k * 48:k * 48 + 48].max()); c = att if v > y else rel
        y = v + (y - v) * (c ** 48); env[k] = y
    lv = 20 * np.log10(env + 1e-9)
    gr = np.where(lv > thr_db, (lv - thr_db) * (1 - 1 / ratio), 0.0)
    return (x * _smooth_gain(10 ** (-gr / 20.0), len(x))[:, None]).astype(np.float32)


def limit(x, ceil_db, look_s=0.005, rel_s=0.08):
    """前瞻峰值限制：块增益取前后 look_s 内所需的最小值，按 rel_s 放回，再逐样本插值（look ≥ 1 块，插值后仍不超过所需）。返回 (输出, 增益曲线)。"""
    n = len(x)
    pk = np.abs(x).max(1)
    need = np.minimum(1.0, 10 ** (ceil_db / 20.0) / np.maximum(pk, 1e-6))
    look = max(48, int(look_s * SR))
    nb = (n + 47) // 48
    gb = np.empty(nb, np.float64); y = 1.0; rr = math.exp(-1 / (rel_s * SR))
    for k in range(nb):
        i0 = k * 48
        x_ = float(need[max(0, i0 - look):i0 + 48 + look].min())
        y = x_ if x_ < y else x_ + (y - x_) * (rr ** 48)
        gb[k] = y
    g = _smooth_gain(gb, n)
    return x * g[:, None], g


def level(x, tgt, ceil, max_gr=4.0, comp=None):
    """把一段声音放到目标电平 tgt（有声 RMS），峰值不过 ceil：可选温和压缩（comp = (阈值相对 tgt, 比例)）→ 增益 → 平滑前瞻限制；
    限制量超过 max_gr 就把目标往下让（每次 1 dB，最多让 6 dB），不靠硬压（用户："角色的语音有些失真"）。返回 (y, 实际电平, 限制量)。"""
    for k in range(7):
        t = tgt - k
        y = x * np.float32(10 ** ((t - active_rms_db(x)) / 20.0))
        if comp:
            y = compress(y, thr_db=t + comp[0], ratio=comp[1], att_s=0.006, rel_s=0.16)
            y = y * np.float32(10 ** ((t - active_rms_db(y)) / 20.0))
        y2, g = limit(y, ceil, look_s=0.005, rel_s=0.15)
        gr = -20 * math.log10(max(1e-6, float(g.min())))
        if gr <= max_gr or k == 6:
            return y2.astype(np.float32), active_rms_db(y2), gr
    return y2.astype(np.float32), active_rms_db(y2), gr


def reverb_ir(rt60, sr=SR, seed=7, damp_hz=5000.0):
    """算法混响冲激响应：左右独立白噪声 × 指数衰减（RT60），低通阻尼，能量归一。返回 [L, 2]。"""
    n = int(rt60 * 1.1 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    ir = rng.normal(0, 1, (n, 2)).astype(np.float32) * np.exp(-6.91 * t / rt60)[:, None].astype(np.float32)
    ir = sosfilt(butter(2, damp_hz, btype="lowpass", fs=sr, output="sos"), ir, axis=0).astype(np.float32)
    ir[:int(0.012 * sr)] *= np.linspace(0, 1, int(0.012 * sr))[:, None]   # 12 ms 预延迟般的软起
    return ir / np.sqrt((ir ** 2).sum(0, keepdims=True) + 1e-9)


def _conv(x, ir):
    from scipy.signal import fftconvolve
    return np.stack([fftconvolve(x[:, c], ir[:, c]) for c in range(2)], 1).astype(np.float32)


def lp_sweep(x, f0, f1, blk=256):
    """时变低通（二阶巴特沃斯，截止频率按对数从 f0 滑到 f1；分块换系数、滤波器状态延续）。"""
    from scipy.signal import lfilter_zi
    out = np.empty_like(x); n = len(x); zi = None
    for i in range(0, n, blk):
        u = i / max(1, n - 1)
        fc = math.exp(math.log(f0) + (math.log(f1) - math.log(f0)) * u)
        b, a = butter(2, min(fc, SR * 0.45), btype="lowpass", fs=SR)
        if zi is None:
            zi = np.stack([lfilter_zi(b, a) * x[0, c] for c in range(2)], 1)
        seg, zi = lfilter(b, a, x[i:i + blk], axis=0, zi=zi)
        out[i:i + blk] = seg
    return out.astype(np.float32)


def recede(a, i_sweep, f0=12000.0, f1=500.0, rt60=3.0, wet=0.6):
    """序章 → 标题卡（用户："完全去掉素材原声的音乐又显得人物干声很粗糙，但是保留音乐会和歌曲衔接有些突兀"）：
    原声从 i_sweep 起低通扫走（像退到远处）、同时混响逐渐长出来；干声在段尾淡完，混响尾巴继续衰减到歌进来之后。返回加长后的数组。"""
    n = len(a)
    dry = a.copy()
    dry[i_sweep:] = lp_sweep(a[i_sweep:], f0, f1)
    ir = reverb_ir(rt60)
    send = np.zeros(n, np.float32)
    send[i_sweep:] = np.sqrt(_ss(np.linspace(0, 1, n - i_sweep)))
    w = _conv(dry * send[:, None], ir) * wet
    w = sosfilt(butter(2, 160, btype="highpass", fs=SR, output="sos"), w, axis=0).astype(np.float32)
    out = w.copy()
    out[:n] += dry
    return out


def song_intro(x, hp_hz=500.0, hp_s=1.2, swell_s=1.4, swell_db=-8.0):
    """歌开头没有前奏、第一拍直接重低音砸进来（0.2 s 处 -21 dB，80% 能量 < 300 Hz）→
    ① 第一拍前加"反向混响"：取第一拍 0.5 s 做混响、只要湿声、倒放，正好在第一拍上收住（从无声升起来，第一拍变成"落地"）；
    ② 头 hp_s 秒从高通版渐变回全频（低音逐渐进来）。返回 (x2, 前置长度)，x2 = 反向混响前置 + 处理后的歌。"""
    on = int(np.argmax(np.abs(x).max(1) > 0.05))
    hit = x[on:on + int(0.5 * SR)]
    wet = _conv(hit, reverb_ir(2.2, seed=11))[:int(swell_s * SR)]
    sw = wet[::-1].copy()
    sw *= np.sqrt(_ss(np.linspace(0, 1, len(sw))))[:, None]
    sw = sosfilt(butter(2, 140, btype="highpass", fs=SR, output="sos"), sw, axis=0).astype(np.float32)
    sw *= np.float32(10 ** ((rms_at(x, on, on + int(0.5 * SR)) + swell_db - rms_at(sw, len(sw) - int(0.4 * SR), len(sw))) / 20.0))
    y = x.copy()
    m = on + int(hp_s * SR)
    hp = sosfilt(butter(4, hp_hz, btype="highpass", fs=SR, output="sos"), x[:m], axis=0).astype(np.float32)
    u = np.zeros(m, np.float32); u[on:] = 0.35 + 0.65 * _ss(np.linspace(0, 1, m - on)); u = u[:, None]   # 第一拍带着 1/3 的低音落地，再在 hp_s 内全开
    y[:m] = hp * (1 - u) + x[:m] * u
    pre = max(0, len(sw) - on)                              # 反向混响的末尾对齐第一拍
    out = np.zeros((pre + len(y), 2), np.float32)
    out[pre:] = y
    out[pre + on - len(sw):pre + on] += sw
    return out, pre / SR


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
    on = e[e > e.max() * 0.15] if e.size else e
    return 20 * np.log10(np.sqrt((on ** 2).mean()) + 1e-9) if on.size else -60.0


def _ss(t):
    return t * t * (3 - 2 * t)


def env(L, fi, fo):
    """淡入 / 淡出包络：等功率 S 曲线 sqrt(smoothstep)（两段交叉时功率和恒定，起止斜率平缓）。"""
    e = np.ones(L, np.float32)
    fi, fo = min(int(fi * SR), L // 2), min(int(fo * SR), L // 2)
    if fi:
        e[:fi] = np.sqrt(_ss(np.linspace(0, 1, fi)))
    if fo:
        e[L - fo:] = np.sqrt(_ss(np.linspace(1, 0, fo)))
    return e


def load_song():
    wav = [q for q in glob.glob(r"D:\备份\*.wav") if "PROVANT" in os.path.basename(q)][0]
    r = subprocess.run([FF, "-v", "error", "-i", wav, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2).copy()


# 原片配乐压低量：环境底 / 爆点音效是垫在歌下面的短片段，里面的原片配乐每个切点一开一关（"断断续续"），压到 -18 dB（残留被歌盖住）；
# 台词 / 原声段只压一点（完全去配乐会让人声失真，用户 10-20）
BED_BG, FX_BG = -18.0, -18.0
CM_BED = {"OKAteito4K6": [(23.8, 30.1)], "OKjiend4K60": [(15.6, 23.1)], "OKyamatai4K": [(28.0, 38.1)]}   # 片尾旁白念标题的区间不取


def tape_stop(x, L):
    """磁带停转：最后 L 秒按线性减速读源（1 → 0），音高随之下滑；末 80 ms 再淡出。"""
    n = int(L * SR)
    if len(x) <= n:
        return x
    head, tail = x[:-n], x[-n:]
    t = np.arange(n, dtype=np.float64)
    pos = t - t * t / (2.0 * n)                             # ∫(1 - t/n)
    out = np.stack([np.interp(pos, np.arange(n), tail[:, c]) for c in range(2)], 1).astype(np.float32)
    out *= env(n, 0.0, 0.08)[:, None]
    return np.concatenate([head, out])


def rms_at(x, i0, i1):
    return 20 * np.log10(np.sqrt((x[max(0, i0):max(i0 + 1, i1)] ** 2).mean()) + 1e-9)


def _free(cl, a, b):
    """[a, b) 去掉已占用区间 cl 后剩下的段。"""
    out = [(a, b)]
    for c0, c1 in cl:
        nxt = []
        for x0, x1 in out:
            if c1 <= x0 or c0 >= x1:
                nxt.append((x0, x1))
            else:
                if x0 < c0:
                    nxt.append((x0, c0))
                if c1 < x1:
                    nxt.append((c1, x1))
        out = nxt
    return [(x0, x1) for x0, x1 in out if x1 - x0 >= 0.3]


def bed_reads(n):
    """环境底要读的源片段：[{yid, src, dur, a, b, shot}]（a = 成片起点）。
    同一段源声音全片只出现一次（用户："冲田总司的片段语音存在重复 0:00:43–0:00:47"）：
      1) 台词用掉的源区间（±0.15 s）、原声段用掉的源区间、冲田 CM 片尾旁白区间先占住；
      2) 原速镜头（0.85–1.15×）按画面同步取自己的源区间，跳过已占的部分；
      3) 慢放的冲田 CM 镜头（画面和声音本来就不同步）：同源、成片首尾相接的一串接着往下读，跳过已占的源区间，凑够镜头时长。"""
    dl = EDL.get("dialog", [])
    cl = {}
    for d in dl:
        cl.setdefault(d["yid"], []).append((d["s0"] - 0.15, d["s1"] + 0.15))
    for y, rs in CM_BED.items():
        cl.setdefault(y, []).extend(rs)
    for d in EDL.get("native_audio", []):                  # 原声段（break / 序章 / 器乐段）用掉的源区间
        cl.setdefault(d["yid"], []).append((d["src"] - 0.1, d["src"] + d["t1"] - d["t0"] + 0.1))
    cand = []
    for s in EDL["shots"]:
        cm = s["yid"] in CM_BED                              # 冲田 CM（用户："冲田总司片段的原音包括语音和环境音请补齐"）：慢放镜头也按原速取环境声
        sync = 0.85 <= s.get("speed", 1) <= 1.15
        if (s["yid"] not in V2.FILM and not cm) or s.get("hold") or (not cm and not sync) or s["b"] - s["a"] < 0.4:
            continue
        cand.append((s, sync))
    out = []
    for s, sync in cand:                                    # 2) 原速：同步
        if not sync:
            continue
        y, src, dur = s["yid"], s["src_a"], s["b"] - s["a"]
        for p0, p1 in _free(cl.get(y, []), src, src + dur):
            out.append(dict(yid=y, src=round(p0, 4), dur=round(p1 - p0, 4), a=s["a"] + (p0 - src), shot=s["id"]))
        cl.setdefault(y, []).append((src, src + dur))
    cur, last = {}, {}
    for s, sync in cand:                                    # 3) 慢放：接着读
        if sync:
            continue
        y, dur = s["yid"], s["b"] - s["a"]
        c = cur[y] if (y in last and abs(last[y] - s["a"]) < 0.02) else s["src_a"]
        t, need = s["a"], dur
        for p0, p1 in _free(cl.get(y, []), c, c + dur * 4 + 10):
            if need < 0.3:
                break
            take = min(p1 - p0, need)
            out.append(dict(yid=y, src=round(p0, 4), dur=round(take, 4), a=t, shot=s["id"]))
            cl.setdefault(y, []).append((p0, p0 + take))
            t += take; need -= take; c = p0 + take
        cur[y], last[y] = c, s["b"]
    for r in out:
        r["b"] = r["a"] + r["dur"]
    return sorted(out, key=lambda r: r["a"])


def main():
    n = int(round(EDL["T1"] * SR))
    song = load_song() * np.float32(10 ** (-4.0 / 20))      # v9：环境音和音乐持平，叠起来 +3 dB → 音乐先让 4 dB
    music = np.zeros((n, 2), np.float32)
    segs = EDL["song_map"]
    xf = EDL.get("song_xf") or [{} for _ in segs]
    for k, (f0, a, b) in enumerate(segs):                   # 用户："将音乐淡出和渐入做的自然一些"：停歌处 fo 秒淡出、回歌提前 pre 秒交叉淡入
        pre = xf[k].get("pre", 0.0) if k else 0.0
        fo = xf[k].get("fo", 0.3) if k < len(segs) - 1 else 0.0
        x = song[int((a - pre) * SR):int(b * SR)].copy()
        x *= env(len(x), max(pre, 0.02) if k else 0.0, fo)[:, None]
        if k == 0 and xf[0].get("intro"):                   # 歌的入场：反向混响 + 低音渐入
            x, pre = song_intro(x, **xf[0]["intro"])
            print("歌入场：反向混响前置 %.2f s" % pre)
        i0 = int(round((f0 - pre) * SR)); i1 = min(n, i0 + len(x))
        music[i0:i1] += x[:i1 - i0]
    ev = [dict(e, gain=e["gain"] - 4.0) for e in V2.events() if e["kind"] in ("hit", "rise", "accent")]
    reads = bed_reads(n)
    SB.prefetch([(d["yid"], d["src"], d["t1"] - d["t0"]) for d in EDL.get("native_audio", [])] +
                [(e["yid"], e["src"], e["dur"]) for e in ev] + [(r["yid"], r["src"], r["dur"]) for r in reads])
    nat = np.zeros_like(music); spans = []
    for d in EDL.get("native_audio", []):                  # 前奏 / 间奏 / 尾声：原片整轨
        # 用户 10-20（看过 v10）："素材原声不用单独提取环境音和人声（这样会导致失真），将原声的背景音压低一点，羽蛇神那一段…就按原声放出就行"
        raw = d.get("audio") == "raw"
        a = V2.audio(d["yid"], d["src"], d["t1"] - d["t0"]) if raw else SB.bg(d["yid"], d["src"], d["t1"] - d["t0"], d.get("bg_db", -6.0))
        if a is None:
            print("! native 取不到", d); continue
        ar0 = active_rms_db(a)
        if raw:                                             # 魁札尔：原样，只做静态增益（峰值封顶 -1.5）
            g = min(d["level"] - ar0, -1.5 - 20 * math.log10(float(np.abs(a).max()) + 1e-9))
            a = a * np.float32(10 ** (g / 20.0)); d["_eff"], d["_gr"] = ar0 + g, 0.0
        else:                                               # 其余：平滑前瞻限制最多 4 dB，不够就让目标
            a, d["_eff"], d["_gr"] = level(a, d["level"], -1.5, max_gr=4.0)
            g = d["_eff"] - ar0
        if d.get("recede"):                                 # 序章原声退远 + 混响尾巴（干声在段尾淡完，尾巴越过段尾）
            rc = d["recede"]
            a = a * env(len(a), d.get("fi", 0.3), d.get("fo", 0.6))[:, None]
            a = recede(a, int((rc["t"] - d["t0"]) * SR), rc.get("f0", 12000.0), rc.get("f1", 500.0), rc.get("rt60", 3.0), rc.get("wet", 0.6))
        else:
            a = a * env(len(a), d.get("fi", 0.3), d.get("fo", 0.6))[:, None]
        i0 = int(round(d["t0"] * SR)); i1 = min(n, i0 + len(a))
        nat[i0:i1] += a[:i1 - i0]; spans.append((d["t0"], d["t1"]))
        print("native  %7.2f–%7.2f  %-12s @%7.2f  %s  %+5.1f dB → %.1f dBFS（目标 %.0f），限制 %.1f dB" % (d["t0"], d["t1"], d["yid"], d["src"],
              "原样" if raw else "背景 %+.0f dB" % d.get("bg_db", -6.0), g, d["_eff"], d["level"], d["_gr"]))
    nat, gn = limit(nat, -1.5, look_s=0.005, rel_s=0.12)  # 原声总线只当保险（各段已按峰值封顶）
    print("原声总线限制：最大 %.2f dB" % (20 * math.log10(max(1e-6, float(gn.min())))))
    ev = [e for e in ev if not any(e["t"] < a1 and e["t"] + e["dur"] > a0 - 0.2 for a0, a1 in spans)]   # 不和原声段重叠（换气 rise 会预放 break 开头）
    fx = np.zeros_like(music)
    for e in ev:
        a = SB.bg(e["yid"], e["src"], e["dur"], FX_BG)
        if a is None:
            continue
        L = len(a)
        a = a * env(L, e.get("fade_in", 0.02), min(0.25, e["dur"] / 3))[:, None]
        i0 = int(e["t"] * SR); i1 = min(n, i0 + L)
        if rms_at(music, i0, i1) > -11.0:
            e["gain"] = min(e["gain"], -18.0)
        fx[i0:i1] += (a * (10 ** (e["gain"] / 20.0)))[:i1 - i0]
    dl = EDL.get("dialog", [])
    busy = spans + [(d["t"] - 0.3, d["t"] + (d["s1"] - d["s0"]) + 0.3) for d in dl] + [(e["t"] - 0.2, e["t"] + e["dur"]) for e in ev]   # 台词底下不垫环境底（去配乐后环境底里也有人声，会和台词叠成两句）
    hp = butter(2, 150, btype="highpass", fs=SR, output="sos")
    bed = np.zeros_like(music); nb = 0; items = []; skipped = 0
    for r in reads:                                         # 环境底（用户："可以加上一些原片的环境音"）
        if any(a0 < r["b"] and r["a"] < a1 for a0, a1 in busy):
            continue
        a = SB.bg(r["yid"], r["src"], r["dur"], BED_BG)
        o = SB.bg(r["yid"], r["src"], r["dur"], 0.0)
        if a is None or o is None or np.abs(a).max() < 1e-3:
            continue
        if active_rms_db(a) < active_rms_db(o) - 9.0:      # 这段原声基本只有配乐：不垫（去掉配乐后只剩残渣）
            skipped += 1; continue
        a = sosfilt(hp, a, axis=0).astype(np.float32)
        i0 = int(round(r["a"] * SR)); i1 = min(n, i0 + len(a))
        tgt = max(-32.0, rms_at(music, i0, i1) - 5.0)     # 用户："环境音稍微小一点…有些盖过了音乐"：比所在位置的音乐低 5 dB
        a = a * np.float32(10 ** (min(12.0, tgt - active_rms_db(a)) / 20.0))
        a = compress(a, thr_db=tgt + 2.0, ratio=2.0, att_s=0.008, rel_s=0.20)   # 只削爆点（平滑增益）
        a = a * np.float32(10 ** (min(3.0, tgt - active_rms_db(a)) / 20.0))
        items.append([i0, a])
    for k, (i0, a) in enumerate(items):
        prev_end = items[k - 1][0] + len(items[k - 1][1]) if k else -10 ** 9
        nxt_start = items[k + 1][0] if k + 1 < len(items) else 10 ** 9
        fi = 0.08 if i0 - prev_end < int(0.05 * SR) else 0.30
        fo = 0.08 if nxt_start - (i0 + len(a)) < int(0.05 * SR) else 0.30
        a = a * env(len(a), fi, fo)[:, None]
        i1 = min(n, i0 + len(a))
        bed[i0:i1] += a[:i1 - i0]; nb += 1
    print("环境底：用 %d 段，%d 段基本只有配乐、跳过" % (nb, skipped))
    bed, _ = limit(bed, -7.0, look_s=0.004, rel_s=0.08)   # 环境底总线峰值 -7 dBFS
    print("环境底镜头: %d" % nb)
    dbus = np.zeros_like(music)
    for d in dl:
        i0 = int(d["t"] * SR)
        mus = rms_at(music + nat + bed, i0, i0 + int((d["s1"] - d["s0"]) * SR))
        bgd = -12.0 if mus > -16.0 else -6.0                # 压在歌上的台词，原片配乐多压一点（不然配乐碎片跟着台词一闪一闪）；歌停了就只压一点
        if d.get("voice") == "sep":                         # 贞德 / 马修：v7 的 MDX 分离干声（用户："参考 V7，去掉原声的音乐，用清晰的人声"）
            tail = 0.35                                     # 句尾带上干声里自己的混响余音（不在 0.15 s 里一刀淡完），但不读进下一句
            nxt = [e["s0"] for e in dl if e["yid"] == d["yid"] and e["s0"] > d["s0"]]
            if nxt:
                tail = max(0.0, min(tail, min(nxt) - d["s1"] - 0.05))
            x = eq_voice(stem(d["yid"], d["s0"], d["s1"] + tail), sep=True)
            x = x * env(len(x), 0.03, 0.15 + tail)[:, None]
            bgd = None
        else:
            x = SB.bg(d["yid"], d["s0"], d["s1"] - d["s0"], bgd)
            x = eq_voice(x) * env(len(x), 0.06, 0.22)[:, None]
        i1 = min(n, i0 + len(x))
        tgt = float(np.clip(mus + 6.0, -16.0, -10.0))
        a0 = active_rms_db(x)
        x, eff, gr = level(x, tgt, -2.5, max_gr=5.0, comp=(-2.0, 2.5))   # 2.5:1 慢起压缩 + 平滑限制 ≤ 5 dB，峰值 -2.5
        g = eff - a0
        dbus[i0:i1] += x[:i1 - i0]
        ev.append(dict(kind="dialog", t=d["t"], yid=d["yid"], src=d["s0"], dur=round(d["s1"] - d["s0"], 3), gain=round(float(g), 1),
                       music_db=round(float(mus), 1), target_db=round(float(eff), 1), bg_db=bgd, lim_db=round(gr, 1)))
    dbus, gd = limit(dbus, -2.5, look_s=0.005, rel_s=0.12)
    print("台词总线限制：最大 %.2f dB" % (20 * math.log10(max(1e-6, float(gd.min())))))
    music = carve(music, dbus, depth_db=8.0)               # 台词变温和了，中频让位多一点（只挖 M 声道 250–4500 Hz，听感上音乐不变小）
    if os.environ.get("DUMP_STEMS"):                       # 诊断：各总线单独存一份
        np.savez(os.path.join(OUT, "stems_%s.npz" % VER), music=music, nat=nat, fx=fx, bed=bed, dbus=dbus)
    out, g = limit(music + nat + fx + bed + dbus, -1.0, look_s=0.005, rel_s=0.10)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "mix_%s.wav" % VER)
    subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_s24le", p],
                   input=out.astype(np.float32).tobytes(), check=True)
    ev.sort(key=lambda e: e["t"])
    json.dump(ev, open(os.path.join(ROOT, "data", "audio_events_%s.json" % VER), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for e in ev:
        print("%-7s %7.3f  %-12s @%7.2f  %.2fs  %+5.1fdB%s" % (e["kind"], e["t"], e["yid"], e["src"], e["dur"], e["gain"],
                                                         ("  music %.1f -> voice %.1f dBFS, 限制 %.1f dB" % (e["music_db"], e["target_db"], e["lim_db"])) if "music_db" in e else ""))
    print("mix ->", p, "peak %.3f, limiter min gain %.2f dB @ %.2f s, >1 dB GR: %.2f s total" % (
        np.abs(out).max(), 20 * math.log10(max(1e-6, g.min())), g.argmin() / SR, (g < 10 ** (-1 / 20)).sum() / SR))
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

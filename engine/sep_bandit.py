# -*- coding: utf-8 -*-
r"""
去配乐（用户 10-20："能不能将素材里的音乐和环境音人物音剥离？仅保留人物音和环境音"）。
模型：BandIt v2 multilingual（DnR v3：speech / music / effects）的 music 分支，ONNX（models/bandit，官方权重 Zenodo 12701995，
CC BY-SA 4.0；转换 promptwhat/bgm-separator，哈希已核对）。非音乐 = 原声 × 维纳软掩码（见 Bandit.nomusic）= 人声 + 环境 / 音效。
约定（模型说明）：48 kHz 单声道 8 s 块（384000 样本）；STFT n_fft 2048 / hop 512 / 周期 Hann / center reflect，**不归一化**；
音乐谱 = STFT × 复数掩码，非归一化 iSTFT。双声道：L / R 各算一次掩码。
块间：8 s 窗、HOP_S 步长、Hann 交叠相加。
用法：v4_audio 先 SB.prefetch([(yid, t0, dur)…]) 再用 SB.audio 代替 V2.audio；单测：.venv_gpu python scripts/sep_bandit.py <yid> <t0> <dur>
缓存：E:\provant_work\audio\nomusic\<yid>__<a>_<b>.wav + index.json（区间两侧各多 4 s）
"""
import json, os, subprocess, sys, time
import numpy as np

ROOT = r"D:\A\provant"
FF = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
MD = os.path.join(ROOT, "models", "bandit")
OUT = r"E:\provant_work\audio\nomusic"
SR, N, NFFT, HOP = 48000, 384000, 2048, 512
HOP_S = float(os.environ.get("BANDIT_HOP", "2.0"))
WB = 1.5                                                    # 维纳软掩码里音乐的权重（合成测试里 1.5 最均衡）
WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32)     # 周期 Hann


def stft(x):
    p = np.pad(x, (NFFT // 2, NFFT // 2), mode="reflect")
    n = 1 + (len(p) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(p[idx] * WIN, axis=1).T.astype(np.complex64)           # [1025, frames]


def istft(S, length):
    fr = np.fft.irfft(S.T, n=NFFT, axis=1) * WIN
    n = fr.shape[0]
    out = np.zeros(NFFT + HOP * (n - 1), np.float32); wsum = np.zeros_like(out)
    for k in range(n):
        out[k * HOP:k * HOP + NFFT] += fr[k]; wsum[k * HOP:k * HOP + NFFT] += WIN ** 2
    out = out / np.maximum(wsum, 1e-8)
    return out[NFFT // 2:NFFT // 2 + length]


class Bandit:
    def __init__(self):
        import onnxruntime as ort
        prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.s = [ort.InferenceSession(os.path.join(MD, f), so, providers=prov) for f in ("bandit-split.onnx", "bandit-tf.onnx", "bandit-mask.onnx")]

    def mask(self, x):                                      # 单声道 [N] → (STFT, 复数音乐掩码)
        X = stft(x)
        z = self.s[0].run(None, {self.s[0].get_inputs()[0].name: X.real[None].astype(np.float32),
                                 self.s[0].get_inputs()[1].name: X.imag[None].astype(np.float32)})[0]
        q = self.s[1].run(None, {self.s[1].get_inputs()[0].name: z})[0]
        mr, mi = self.s[2].run(None, {self.s[2].get_inputs()[0].name: q})
        return X, (mr[0] + 1j * mi[0]).astype(np.complex64)

    def music(self, chunk):                                 # chunk [2, N] → 音乐估计 [2, N]（中声道掩码；只做诊断用）
        X, M = self.mask(chunk.mean(0))
        return np.stack([istft(stft(chunk[c]) * M, N) for c in range(2)])

    def nomusic(self, chunk):
        """chunk [2, N] → 去配乐 [2, N]。逐声道算掩码，再按幅度做维纳软掩码：w = |X−MU|² / (|X−MU|² + β|MU|²)，MU = X·M。
        直接 "原声 − 音乐估计" 会被复数掩码的相位误差拖累（合成测试：歌 + 冲田干声等响，SDR 3.3 / 1.6 / 0.0 dB）；
        维纳软掩码 8.9 / 7.4 / 6.5 dB，干声比音乐低 8 dB 时 5.5 dB（β = 1.5，power 2）。"""
        out = []
        for c in range(2):
            X, M = self.mask(chunk[c])
            MU = X * M
            a, b = np.abs(X - MU) ** 2, np.abs(MU) ** 2
            out.append(istft(X * (a / (a + WB * b + 1e-12)), N))
        return np.stack(out)


def load(path, t0=None, dur=None):
    cmd = [FF, "-v", "error"] + (["-ss", "%.3f" % t0] if t0 is not None else []) + (["-t", "%.3f" % dur] if dur else []) + \
          ["-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    a = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32)
    return a[:a.size // 2 * 2].reshape(-1, 2).T.copy()


def separate(x, model, log=""):
    """x [2, L] → 非音乐 [2, L]（8 s 窗、HOP_S 步长、Hann 交叠相加）。"""
    L = x.shape[1]; hop = int(HOP_S * SR)
    pad = np.pad(x, ((0, 0), (N, N + hop)))
    acc = np.zeros_like(pad); wsum = np.zeros(pad.shape[1], np.float32)
    w = (0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(N) + 0.5) / N)).astype(np.float32)
    starts = list(range(0, pad.shape[1] - N + 1, hop))
    t = time.time()
    for k, s in enumerate(starts):
        seg = pad[:, s:s + N]
        if np.abs(seg).max() < 1e-5:
            wsum[s:s + N] += w
            continue
        acc[:, s:s + N] += model.nomusic(seg) * w; wsum[s:s + N] += w
        if k % 20 == 0:
            print("  %s %d/%d  %.0fs" % (log, k, len(starts), time.time() - t), flush=True)
    return (acc / np.maximum(wsum, 1e-6))[:, N:N + L]


# ---------------------------------------------------------------- 按需缓存：只分离混音实际读到的区间
IDX = os.path.join(OUT, "index.json")
MARGIN = 4.0                                                # 区间两侧多分离 4 s（窗口有上下文、边缘无 Hann 半窗）
SRC = json.load(open(os.path.join(ROOT, "data", "audio_sources.json"), encoding="utf-8"))
_M = None


def model():
    global _M
    if _M is None:
        _M = Bandit()
    return _M


def _index():
    return json.load(open(IDX, encoding="utf-8")) if os.path.exists(IDX) else {}


def _make(y, a, b, idx):
    x = load(SRC[y], a, b - a)
    t = time.time()
    nm = separate(x, model(), y)
    f = "%s__%07.2f_%07.2f.wav" % (y, a, b)
    subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_f32le", os.path.join(OUT, f)],
                   input=np.ascontiguousarray(nm.T).astype(np.float32).tobytes(), check=True)
    got = x.shape[1] / SR
    idx.setdefault(y, []).append([round(a, 3), round(a + got, 3) if got >= b - a - 0.05 else 1e9, f])   # 读到源结尾：这段一直覆盖到结尾
    json.dump(idx, open(IDX, "w", encoding="utf-8"), indent=1)
    print("nomusic %-12s %7.2f–%7.2f  (%.0f s)" % (y, a, a + x.shape[1] / SR, time.time() - t), flush=True)


def prefetch(reqs):
    """reqs: [(yid, t0, dur)] → 每个源合并成几段连续区间，缺的才分离。"""
    os.makedirs(OUT, exist_ok=True)
    by = {}
    for y, t0, d in reqs:
        if y in SRC and d > 0:
            by.setdefault(y, []).append((max(0.0, t0 - MARGIN), t0 + d + MARGIN))
    idx = _index()
    for y, rs in sorted(by.items()):
        rs.sort(); mg = []
        for a, b in rs:
            if mg and a <= mg[-1][1] + 8.0:
                mg[-1][1] = max(mg[-1][1], b)
            else:
                mg.append([a, b])
        for a, b in mg:
            if not any(r[0] <= a + 1e-3 and b - 1e-3 <= r[1] for r in idx.get(y, [])):
                _make(y, a, b, idx)


def audio(yid, t0, dur, sr=SR, ch=2):
    """V2.audio 的去配乐版（同签名，返回 [L, 2] 或 None）。"""
    if yid not in SRC or dur <= 0:
        return None
    t0 = max(0.0, t0)
    for _ in range(2):
        for a, b, f in _index().get(yid, []):
            if a <= t0 + 1e-3 and t0 + dur <= b + 1e-3:
                x = load(os.path.join(OUT, f), t0 - a, dur)
                return x.T.copy() if x.shape[1] >= 64 else None
        prefetch([(yid, t0, dur)])
    return None


def _orig(y, a, b, f):
    """去配乐缓存段对应的原声（同一次 ffmpeg 解码、同起点 → 和去配乐版逐样本对齐，相加不会梳状滤波）。缓存成 *_orig.wav。"""
    o = os.path.join(OUT, f.replace(".wav", "_orig.wav"))
    if not os.path.exists(o):
        x = load(SRC[y], a, (b - a) if b < 1e8 else None)
        subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_f32le", o],
                       input=np.ascontiguousarray(x.T).astype(np.float32).tobytes(), check=True)
    return o


def bg(yid, t0, dur, bg_db=-6.0):
    """原声、背景音（配乐）压低 bg_db（用户 10-20："素材原声不用单独提取环境音和人声（这样会导致失真），将原声的背景音压低一点"）。
    维纳掩码是线性的：X·(w + (1−w)·g) = (1−g)·去配乐 + g·原声，所以直接用缓存的去配乐段和同源同起点的原声段线性混合。
    合成测试：背景 −6 dB 时残留失真比完全去配乐低 ~7 dB（13.5–15.9 dB vs 6.5–8.9 dB）。返回 [L, 2]。"""
    if yid not in SRC or dur <= 0:
        return None
    t0 = max(0.0, t0)
    g = 10 ** (bg_db / 20.0)
    for _ in range(2):
        for a, b, f in _index().get(yid, []):
            if a <= t0 + 1e-3 and t0 + dur <= b + 1e-3:
                nm = load(os.path.join(OUT, f), t0 - a, dur)
                og = load(_orig(yid, a, b, f), t0 - a, dur)
                L = min(nm.shape[1], og.shape[1])
                if L < 64:
                    return None
                return ((1 - g) * nm[:, :L] + g * og[:, :L]).T.copy()
        prefetch([(yid, t0, dur)])
    return None


if __name__ == "__main__":
    t0, d = float(sys.argv[2]), float(sys.argv[3])
    prefetch([(sys.argv[1], t0, d)])

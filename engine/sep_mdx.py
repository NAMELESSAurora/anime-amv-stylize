# -*- coding: utf-8 -*-
"""
人声分离（UVR MDX-NET Voc_FT，onnx；.venv_gpu 下走 DirectML 显卡）。
用途：① 歌曲干声 → 逐词起音（歌词跟拍，用户 10-19 第 1 条）；② 片源台词去背景（第 8 条：台词插入很浑浊）。
参数（UVR 模型表，Voc_FT）：n_fft 7680 / hop 1024 / dim_f 3072 / dim_t 256 / compensate 1.021；STFT 与 torch 版一致（hann periodic、center、reflect）。
用法: python scripts/sep_mdx.py IN [OUT_vocals.wav] [--ss S --t T]   （输出 44.1 kHz 立体声 float32 wav）
      from sep_mdx import separate; voc = separate(stereo_float32[N,2])   # 44.1 kHz
"""
import os, subprocess, sys
import numpy as np
import librosa

ROOT = r"D:\A\provant"
FF = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
MODEL = os.path.join(ROOT, "models", "uvr", "models", "MDXNet", "UVR-MDX-NET-Voc_FT.onnx")
SR = 44100
N_FFT, HOP, DIM_F, DIM_T, COMP = 7680, 1024, 3072, 256, 1.021
CHUNK = HOP * (DIM_T - 1)
TRIM = N_FFT // 2
GEN = CHUNK - 2 * TRIM
NB = N_FFT // 2 + 1
_S = None


def sess():
    global _S
    if _S is None:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        prov = ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in ort.get_available_providers():
            so.enable_mem_pattern = False; so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
        _S = ort.InferenceSession(MODEL, so, providers=prov)
    return _S


def _spec(wave):                      # wave [2, CHUNK] → [1, 4, DIM_F, DIM_T]
    out = []
    for ch in range(2):
        X = librosa.stft(wave[ch], n_fft=N_FFT, hop_length=HOP, window="hann", center=True, pad_mode="reflect")
        out += [X.real, X.imag]
    return np.stack(out)[None, :, :DIM_F, :DIM_T].astype(np.float32)


def _ispec(spec):                     # [4, DIM_F, DIM_T] → [2, CHUNK]
    full = np.zeros((4, NB, DIM_T), np.float32); full[:, :DIM_F] = spec
    return np.stack([librosa.istft(full[2 * c] + 1j * full[2 * c + 1], n_fft=N_FFT, hop_length=HOP, window="hann",
                                   center=True, length=CHUNK) for c in range(2)])


def separate(mix):
    """mix: float32 [N, 2] @44.1k → vocals [N, 2]"""
    x = mix.T.astype(np.float32)
    n = x.shape[1]
    pad = GEN - n % GEN
    xp = np.concatenate([np.zeros((2, TRIM), np.float32), x, np.zeros((2, pad), np.float32), np.zeros((2, TRIM), np.float32)], 1)
    s = sess(); name = s.get_inputs()[0].name
    outs = []
    for i in range(0, n + pad, GEN):
        w = xp[:, i:i + CHUNK]
        y = s.run(None, {name: _spec(w)})[0][0]
        outs.append(_ispec(y)[:, TRIM:-TRIM])
    v = np.concatenate(outs, 1)[:, :n] * COMP
    return v.T


def load(path, ss=None, t=None):
    cmd = [FF, "-v", "error"] + (["-ss", "%.3f" % ss] if ss is not None else []) + (["-t", "%.3f" % t] if t is not None else []) + \
          ["-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    a = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32)
    return a[:a.size // 2 * 2].reshape(-1, 2).copy()


def save(path, a, sr=SR):
    subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-", "-c:a", "pcm_f32le", path],
                   input=a.astype(np.float32).tobytes(), check=True)


if __name__ == "__main__":
    import time
    a = sys.argv[1:]
    ss = float(a[a.index("--ss") + 1]) if "--ss" in a else None
    t = float(a[a.index("--t") + 1]) if "--t" in a else None
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
    src = pos[0]; out = pos[1] if len(pos) > 1 else os.path.splitext(src)[0] + "_vocals.wav"
    t0 = time.time()
    mix = load(src, ss, t)
    voc = separate(mix)
    save(out, voc)
    print("%s -> %s  %.1fs audio in %.1fs  (%s)" % (src, out, len(mix) / SR, time.time() - t0, sess().get_providers()[0]))

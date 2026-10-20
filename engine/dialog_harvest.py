# -*- coding: utf-8 -*-
"""
台词素材收集（用户 10-19 第 8 条："台词可以多加一些女性的台词，并且现有的台词插入很浑浊……台词声音可以适当大一点"）。
  1. 每个正片 / 战斗剪辑 / CM 源：UVR MDX 分出人声（去 BGM 和大部分音效）→ E:\\provant_work\\audio\\sep\\<yid>.wav
  2. Silero VAD 找语音段（0.5–4.0 s，前后有停顿）
  3. pyin 估基频：中位数 > 185 Hz 记为女声（动画女声多在 220–350 Hz；喊叫的男声也可能偏高，所以再看文本 / 角色）
  4. faster-whisper small（日语）转写：文本 + 置信度，过滤乱识别 / 非语音
输出 data/dialog_candidates.json（每条：yid / t0 / t1 / f0 / female / text / logprob / clarity）+ 每条的干声 wav（E:\\provant_work\\audio\\lines）
用法: .venv_gpu\\Scripts\\python scripts/dialog_harvest.py [yid ...]
"""
import json, os, sys, time
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(__file__))
import sep_mdx as S

ROOT = r"D:\A\provant"
WORK = r"E:\provant_work\audio"
SRC = json.load(open(os.path.join(ROOT, "data", "audio_sources.json"), encoding="utf-8"))
DEFAULT = ["OKjiend4K60", "OKyamatai4K", "OKAteito4K6", "MUkengo4K60",
           "tlKnVXKaZYs", "B5DdMAhWCsE", "jqQkurxOj-Q", "3U3DFMZBddA", "rZOxWc1nGt8", "aT7kG7zdkak", "vUqzHyadZAA",
           "BWiA7y_XNHY", "_2j1V81wuR0", "deONLS7FcVI", "Yuq57lj3LMU", "o1H-NNBjHbE", "Cf9uMIffDTQ", "DlyEbc5AMUw", "EpLKseoUV6Q"]
_VAD = None


def vad(m16):
    global _VAD
    import onnxruntime as ort
    if _VAD is None:
        _VAD = ort.InferenceSession(os.path.join(ROOT, "models", "silero-vad", "onnx", "model.onnx"), providers=["CPUExecutionProvider"])
    st = np.zeros((2, 1, 128), np.float32); ctx = np.zeros(64, np.float32); out = []
    for k in range(0, len(m16) - 512 + 1, 512):
        x = np.concatenate([ctx, m16[k:k + 512]])[None].astype(np.float32)
        p, st = _VAD.run(None, {"input": x, "state": st, "sr": np.array(16000, np.int64)})
        ctx = m16[k + 448:k + 512]
        out.append(float(p[0, 0]))
    return np.array(out)


def segments(pr, step=0.032):
    segs, k = [], 0
    while k < len(pr):
        if pr[k] > 0.6:
            j = k
            while j < len(pr) and pr[j] > 0.35:
                j += 1
            a, b = k * step, j * step
            if 0.5 <= b - a <= 4.0:
                pre = pr[max(0, k - 8):k].mean() if k else 0.0
                post = pr[j:j + 8].mean() if j < len(pr) else 0.0
                segs.append((a, b, float(pr[k:j].mean()), float(max(pre, post))))
            k = j
        k += 1
    return segs


def main(ids):
    from faster_whisper import WhisperModel
    os.makedirs(os.path.join(WORK, "sep"), exist_ok=True); os.makedirs(os.path.join(WORK, "lines"), exist_ok=True)
    wm = WhisperModel("small", device="cpu", compute_type="int8")
    outp = os.path.join(ROOT, "data", "dialog_candidates.json")
    allc = json.load(open(outp, encoding="utf-8")) if os.path.exists(outp) else []
    allc = [c for c in allc if c["yid"] not in ids]
    for y in ids:
        if not SRC.get(y):
            print("no audio source", y); continue
        t0 = time.time()
        sp = os.path.join(WORK, "sep", y + ".wav")
        if os.path.exists(sp):
            voc = S.load(sp)
        else:
            mix = S.load(SRC[y])
            if len(mix) < S.SR:
                print("empty", y); continue
            voc = S.separate(mix); S.save(sp, voc)
        mono = voc.mean(1)
        m16 = librosa.resample(mono, orig_sr=S.SR, target_sr=16000)
        segs = segments(vad(m16))
        n = 0
        for a, b, conf, edge in segs:
            seg = mono[int(a * S.SR):int(b * S.SR)]
            if len(seg) < S.SR * 0.4:
                continue
            f0, vf, _ = librosa.pyin(librosa.resample(seg, orig_sr=S.SR, target_sr=16000), fmin=70, fmax=700, sr=16000, frame_length=1024)
            f0m = float(np.nanmedian(f0[vf])) if np.any(vf) else float("nan")
            segs16 = m16[int(a * 16000):int(b * 16000)]
            res, info = wm.transcribe(segs16.astype(np.float32), language="ja", beam_size=3, vad_filter=False, condition_on_previous_text=False)
            res = list(res)
            text = "".join(r_.text for r_ in res).strip()
            lp = float(np.mean([r_.avg_logprob for r_ in res])) if res else -9.0
            nsp = float(np.mean([r_.no_speech_prob for r_ in res])) if res else 1.0
            rms_v = float(np.sqrt((seg ** 2).mean()))
            c = dict(yid=y, t0=round(a, 3), t1=round(b, 3), dur=round(b - a, 2), vad=round(conf, 2), edge=round(edge, 2),
                     f0=round(f0m, 1) if f0m == f0m else None, female=bool(f0m == f0m and f0m > 185), text=text,
                     logprob=round(lp, 2), no_speech=round(nsp, 2), rms=round(rms_v, 4))
            c["wav"] = os.path.join(WORK, "lines", "%s_%07.3f.wav" % (y, a))
            S.save(c["wav"], voc[max(0, int((a - 0.08) * S.SR)):int((b + 0.15) * S.SR)])
            allc.append(c); n += 1
        print("%-12s segs %3d kept %3d  %.0fs" % (y, len(segs), n, time.time() - t0), flush=True)
        json.dump(allc, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("->", outp, len(allc))


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)

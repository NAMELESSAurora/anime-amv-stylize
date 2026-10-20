# -*- coding: utf-8 -*-
"""成片级闪烁预检（v10）：把每个成片帧映射到源帧亮度（160×90 灰度均值，和 qc_v1 同尺度），
R / P 镜头按 r_base 压暗 ×0.78，算 |ΔY| >= 51 的跳变，报告 1 s 窗内 > 3 次的位置和对应镜头。用法：python scripts/timeline_flash.py v10"""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from cards import src_path
ROOT = r"D:\A\provant"
FF = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
FPS = 60
key = sys.argv[1] if len(sys.argv) > 1 else "v10"
c = json.load(open(os.path.join(ROOT, "data", "edl_%s.json" % key), encoding="utf-8"))
N = int(c["T1"] * FPS)
Y = np.zeros(N, np.float32); who = [None] * N
for s in c["shots"]:
    a, b = s["lo"], s["hi"]
    p = subprocess.run([FF, "-v", "error", "-ss", "%.3f" % a, "-t", "%.3f" % (b - a + 0.05), "-i", src_path(s["yid"]), "-vf", "scale=160:90",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    fr = np.frombuffer(p.stdout, np.uint8)
    n = len(fr) // (160 * 90)
    if n == 0:
        print("decode fail", s["id"]); continue
    m = fr[:n * 160 * 90].reshape(n, -1).mean(1)
    fps_src = n / (b - a + 0.05)
    k = 0.78 if s["look"] in ("R", "P") else 1.0
    for F in range(int(round(s["a"] * FPS)), min(N, int(round(s["b"] * FPS)))):
        rel = F / FPS - s["a"]
        if s.get("hold"):
            rel = np.floor(rel * s["hold"]) / s["hold"]
        t = np.clip(s["src_a"] + rel * s["speed"], a, b)
        Y[F] = m[int(np.clip((t - a) * fps_src, 0, n - 1))] * k; who[F] = s["id"] + ":" + s["code"]
d = np.abs(np.diff(Y)) >= 51
bad = []
F = 0
while F < N - 60:
    if d[F:F + 60].sum() > 3:
        js = np.nonzero(d[F:F + 60])[0] + F
        bad.append((F / FPS, [(round(j / FPS, 2), who[j + 1], int(Y[j]), int(Y[j + 1])) for j in js]))
        F += 60
    else:
        F += 1
print("跳变总数", int(d.sum()), "；违规窗", len(bad))
for t, js in bad:
    print("  %.2f s:" % t, js)

# -*- coding: utf-8 -*-
"""
v1 成片 QC：逐帧亮度统计（ffmpeg 缩到 160×90 灰度读裸流）。

检查项（验收定义第 5 条）：
  黑帧     平均亮度 < 4 且 99.5 分位高光 < 24（暗场里有发光体的不算；章节卡均值约 10–20，不会误报）
  闪烁     相邻帧亮度跳变 ≥ 20%（≥ 51 级）记一次"闪"；任意 1 s 滑窗内 > 3 次 = 违规（WCAG 2.3.1 一般闪光阈值的简化版）
  拼接点   分段渲染的接缝前后 3 帧亮度连续
另出 out/v1/qc_sheet_<k>.jpg：每 2 s 抽一帧的拼图（人工看）。

用法: python scripts/qc_v1.py out/v1/PROVANT_AMV_v1_video.mov
"""
import json, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw

FF = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
FPS = 60


def lumas(path, with_hi=False):
    """逐帧亮度均值（和 99.5 分位高光）。"""
    p = subprocess.Popen([FF, "-v", "error", "-i", path, "-vf", "scale=160:90,format=gray", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE)
    out, hi = [], []
    while True:
        b = p.stdout.read(160 * 90)
        if len(b) < 160 * 90:
            break
        a = np.frombuffer(b, np.uint8).astype(np.float32)
        out.append(a.mean()); hi.append(np.percentile(a, 99.5))
    return (np.array(out), np.array(hi)) if with_hi else np.array(out)


def sheets(path, n, outdir, every=2.0, per=48):
    times = np.arange(0, n / FPS, every)
    for k in range(0, len(times), per):
        ts = times[k:k + per]
        im = Image.new("RGB", (8 * 320, ((len(ts) + 7) // 8) * 180))
        d = ImageDraw.Draw(im)
        for i, t in enumerate(ts):
            r = subprocess.run([FF, "-v", "error", "-ss", "%.3f" % t, "-i", path, "-frames:v", "1", "-vf", "scale=320:180",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
            if len(r) >= 320 * 180 * 3:
                im.paste(Image.frombuffer("RGB", (320, 180), r[:320 * 180 * 3]), ((i % 8) * 320, (i // 8) * 180))
            d.rectangle([(i % 8) * 320, (i // 8) * 180, (i % 8) * 320 + 64, (i // 8) * 180 + 16], fill=(0, 0, 0))
            d.text(((i % 8) * 320 + 3, (i // 8) * 180 + 2), "%.1fs" % t, fill=(255, 230, 0))
        im.save(os.path.join(outdir, "qc_sheet_%d.jpg" % (k // per)), quality=82)


def main():
    path = sys.argv[1]
    seams = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else []
    L, HI = lumas(path, with_hi=True)
    n = len(L)
    rep = {"frames": n, "seconds": round(n / FPS, 3), "luma_mean": round(float(L.mean()), 1)}
    # 真黑 = 均值 < 4 且高光 < 24（和 edl_v3.flash_safe 同一定义）。E07 Rider 暗场魔眼均值 3 但眼睛高光 255，不算黑帧
    black = [i for i in range(n) if L[i] < 4 and HI[i] < 24]
    rep["dark_frames"] = int(sum(1 for i in range(n) if L[i] < 4))
    rep["black_frames"] = black[:50]
    rep["black_count"] = len(black)
    jumps = np.abs(np.diff(L)) >= 51
    idx = np.nonzero(jumps)[0] + 1
    viol = []
    for i in range(n):
        c = int(((idx >= i) & (idx < i + FPS)).sum())
        if c > 3:
            viol.append((i, c))
    # 合并连续违规窗口
    spans = []
    for i, c in viol:
        if spans and i <= spans[-1][1] + 1:
            spans[-1][1] = i; spans[-1][2] = max(spans[-1][2], c)
        else:
            spans.append([i, i, c])
    rep["flash_events"] = len(idx)
    rep["flash_violations"] = [{"from_s": round(a / FPS, 2), "to_s": round((b + FPS) / FPS, 2), "max_per_s": c} for a, b, c in spans]
    rep["seams"] = {str(s): [round(float(x), 1) for x in L[max(0, s - 3):s + 3]] for s in seams}
    outdir = os.path.dirname(os.path.abspath(path))
    np.save(os.path.join(outdir, "qc_luma.npy"), L)
    json.dump(rep, open(os.path.join(outdir, "qc_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    sheets(path, n, outdir)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
量每条 conform 源的有效画面区域（去掉 letterbox / pillarbox 黑边）。

为什么：conform 时非 16:9 的源被 pad 成 1920×1080，黑边是画面的一部分。
直接 cover 缩放进格子，黑边会进格子里。成片脚本按这里的有效区域先放大裁掉黑边。

方法：每条源均匀取 15 个时间点跑 cropdetect（limit 24/255），取各点检测框的并集
（并集 = 最大有效区域，避免某个暗场把框判小）。

输出: data/letterbox.json  {yid: {"x":..,"y":..,"w":..,"h":..,"bars": bool}}
"""
import json, os, re, subprocess

FF = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
FP = FF.replace("ffmpeg.exe", "ffprobe.exe")
ROOT = r"D:\A\provant"


def crop_at(path, t):
    p = subprocess.run([FF, "-v", "info", "-ss", "%.2f" % t, "-i", path, "-frames:v", "8", "-an",
                        "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", p.stderr)
    return tuple(map(int, m[-1])) if m else None


def main():
    idx = json.load(open(os.path.join(ROOT, "data", "conform_index.json"), encoding="utf-8"))
    shots_idx = json.load(open(os.path.join(ROOT, "data", "shots", "index.json"), encoding="utf-8"))
    want = {r["yid"] for r in shots_idx}
    out, seen = {}, set()
    for r in idx:
        y = r["yid"]
        if y in seen or y not in want:
            continue
        seen.add(y)
        dur = float(subprocess.run([FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", r["path"]],
                                   capture_output=True, text=True).stdout.strip())
        boxes = [b for b in (crop_at(r["path"], dur * (i + 0.5) / 15) for i in range(15)) if b]
        x0 = min(b[2] for b in boxes); y0 = min(b[3] for b in boxes)
        x1 = max(b[2] + b[0] for b in boxes); y1 = max(b[3] + b[1] for b in boxes)
        bars = (x1 - x0) < 1900 or (y1 - y0) < 1060
        out[y] = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "bars": bars}
        print("%-12s %4d×%-4d @(%d,%d) %s" % (y, x1 - x0, y1 - y0, x0, y0, "LETTERBOX" if bars else ""))
    json.dump(out, open(os.path.join(ROOT, "data", "letterbox.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

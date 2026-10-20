# -*- coding: utf-8 -*-
"""
镜头级素材清单：把每条源切成镜头，量出每个镜头的时长和画面指标，再按镜头出缩略图拼成审片表。

为什么要做：
  sheets2 的接触表每条源只均匀抽 12 帧，抽到的不一定是镜头，排不出镜头表。
  动画镜头多数只有 1–3 s，而本片默认镜头单位是 2 小节（4.66 s），
  所以排片必须知道每个镜头实际有多长、哪些够长、哪些只能放进小格里拼用。

每个镜头量的指标（只看画面中间 74% 高度，避开 letterbox 黑边）：
  yavg   平均亮度 0–255
  ylow   亮度 10% 分位（暗部有多黑）
  dark   ylow < 20 的帧占比 → 漫画化以后容易变成整块墨黑（MANGA_CHAIN_NOTES §七）
  sat    平均饱和度（回色段落挑颜色强的镜头）
  motion 镜头内相邻帧差异的均值（scdet score），越大越动

输出:
  data/shots/<yid>.json          每条源的镜头表
  data/shots/sheets/<yid>_NN.jpg 镜头审片表（每张 6×5 = 30 个镜头，每格标镜号、入点、时长）
  data/shots/index.json          全部源的汇总

用法:
  python shot_inventory.py                 # 默认：SF + FGO 全部源
  python shot_inventory.py --only DlyEbc5AMUw
"""
import argparse, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont

FFDIR = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin"
FF = os.path.join(FFDIR, "ffmpeg.exe")
ROOT = r"D:\A\provant"
OUT = os.path.join(ROOT, "data", "shots")
SHEETS = os.path.join(OUT, "sheets")
FONT = "C:/Windows/Fonts/arial.ttf"

SF = ["5Jyflcob9_k", "EpLKseoUV6Q", "DlyEbc5AMUw", "rHlLzmlc69I", "tcC9g9dlv58"]
HF = ["oZlRdc9gxE4", "cpxwZP1GfAs", "hQcHTLb-FJs", "tY4SZ_NKKb4", "uvLyRoYugbc", "ghwoaF-VYS0",   # 10-18 用户指定的 D:\Fate_AMV\hf 里补进来的
      "vUqzHyadZAA", "vSaILO_7UVk", "tlKnVXKaZYs", "rZOxWc1nGt8", "jqQkurxOj-Q", "aT7kG7zdkak", "a8QQN9RDrSk",
      "9c4ozWgw3Ns", "B5DdMAhWCsE", "3U3DFMZBddA",   # 2026-10-18 用户补的 HF 剧场版片段
      "5IiGdXV-wow", "fKoHcAM0PEk", "5ZLnndSF3rk", "iWpZhkxdcss"]   # v2 补的 HF 4K（Lancer vs Assassin / SA vs Berserker / 两版 HF OP）
# v2（2026-10-18 用户要求素材 2K+）补的其他系列 4K/8K 源
UBW = ["g3n5c5CwDzs", "ViVJDXIBcMk", "fh57udNMKvU"]
APO = ["CQ8v15KMND4", "VobxMYXmkao", "5m13-yHL7EU", "RxmNey6OU_s", "BWiA7y_XNHY", "kgqkkGuyRjU"]
OTHER = {"7CBcMPi6Q84": "HA", "Mmz4LBlaR8E": "HA", "3zV3RgFcyiY": "SR"}   # hollow ataraxia / Samurai Remnant
FGO = ["ONuRaCcA8iw", "62hOiNH0K44", "Yuq57lj3LMU", "qiANLuvPb9o", "_2j1V81wuR0",
       "o1H-NNBjHbE", "kxz8k4ycXCA", "deONLS7FcVI", "PmOY1wzlRjo", "11vAh33_2eI",
       "L02aPFAhTCk", "UuxL121gwN8", "G9nHP6aRsjE", "7R9fdZLTYjo", "Cf9uMIffDTQ"]

SCD_T = 10.0       # scdet 阈值（0–100）。动画硬切的分数一般远高于 10
MIN_SHOT = 0.25    # 短于 6 帧的视为闪切/转场残片，并入前一个镜头
COLS, ROWS = 6, 5
TW, TH = 256, 144


def series_of(yid):
    if yid in SF: return "SF"
    if yid in HF: return "HF"
    if yid in UBW: return "UBW"
    if yid in APO: return "APO"
    return OTHER.get(yid, "FGO")


def src_paths():
    idx = json.load(open(os.path.join(ROOT, "data", "conform_index.json"), encoding="utf-8"))
    out = {}
    for r in idx:
        if r["yid"] not in out and os.path.exists(r["path"]):
            out[r["yid"]] = r["path"]
    return out


def analyse(path):
    """一遍解码：scdet 打切点 + signalstats 逐帧量亮度/饱和度。返回逐帧记录。"""
    vf = ("crop=iw:ih*0.74,scale=320:-2,format=yuv420p,"
          "scdet=threshold=%g:sc_pass=0,signalstats,metadata=mode=print:file=-" % SCD_T)
    # 解码交给 GPU（NVDEC 支持的编码才生效；ProRes 不支持时 ffmpeg 自动回退 CPU）
    p = subprocess.run([FF, "-v", "error", "-hwaccel", "auto", "-i", path, "-an", "-vf", vf, "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    frames, cur = [], None
    for line in p.stdout.splitlines():
        m = re.match(r"frame:(\d+)\s+pts:\S+\s+pts_time:([\d.]+)", line)
        if m:
            cur = {"t": float(m.group(2))}
            frames.append(cur)
            continue
        m = re.match(r"lavfi\.([\w.]+)=([-\d.eE]+)", line)
        if m and cur is not None:
            cur[m.group(1)] = float(m.group(2))
    return frames


def cut_shots(frames, dur):
    cuts = [0.0] + [f["t"] for f in frames if "scd.time" in f] + [dur]
    shots = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < MIN_SHOT and shots:
            shots[-1]["out"] = b
            continue
        shots.append({"in": a, "out": b})
    for i, s in enumerate(shots):
        fs = [f for f in frames if s["in"] <= f["t"] < s["out"]]
        n = max(len(fs), 1)
        ylow = [f.get("signalstats.YLOW", 0) for f in fs]
        s.update({
            "n": i + 1,
            "dur": round(s["out"] - s["in"], 3),
            "in": round(s["in"], 3), "out": round(s["out"], 3),
            "yavg": round(sum(f.get("signalstats.YAVG", 0) for f in fs) / n, 1),
            "ylow": round(sum(ylow) / n, 1),
            "dark": round(sum(1 for v in ylow if v < 20) / n, 2),
            "sat": round(sum(f.get("signalstats.SATAVG", 0) for f in fs) / n, 1),
            "motion": round(sum(f.get("scd.score", 0) for f in fs[1:]) / max(n - 1, 1), 2),
        })
    return shots


def thumb(path, t):
    p = subprocess.run([FF, "-v", "error", "-ss", "%.3f" % t, "-i", path, "-frames:v", "1",
                        "-vf", "scale=%d:%d" % (TW, TH), "-f", "image2pipe", "-vcodec", "png", "-"],
                       capture_output=True)
    from io import BytesIO
    try:
        return Image.open(BytesIO(p.stdout)).convert("RGB")
    except Exception:
        return Image.new("RGB", (TW, TH), (60, 0, 0))


def fmt(t):
    return "%d:%05.2f" % (int(t // 60), t % 60)


def make_sheets(yid, path, shots):
    font = ImageFont.truetype(FONT, 15)
    with ThreadPoolExecutor(4) as ex:
        thumbs = list(ex.map(lambda s: thumb(path, (s["in"] + s["out"]) / 2), shots))
    per = COLS * ROWS
    names = []
    for k in range(0, len(shots), per):
        sheet = Image.new("RGB", (COLS * TW, ROWS * TH), (0, 0, 0))
        d = ImageDraw.Draw(sheet)
        for j, (s, im) in enumerate(zip(shots[k:k + per], thumbs[k:k + per])):
            x, y = (j % COLS) * TW, (j // COLS) * TH
            sheet.paste(im, (x, y))
            label = "S%03d %s +%.1fs" % (s["n"], fmt(s["in"]), s["dur"])
            d.rectangle([x, y, x + 8 * len(label) + 4, y + 18], fill=(0, 0, 0))
            d.text((x + 3, y + 1), label, font=font, fill=(255, 230, 0))
            d.rectangle([x, y, x + TW - 1, y + TH - 1], outline=(90, 90, 90))
        nm = "%s_%02d.jpg" % (yid, k // per + 1)
        sheet.save(os.path.join(SHEETS, nm), quality=82)
        names.append(nm)
    return names


def probe_dur(path):
    p = subprocess.run([FF.replace("ffmpeg.exe", "ffprobe.exe"), "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(p.stdout.strip())


def run_one(yid, path):
    dur = probe_dur(path)
    frames = analyse(path)
    shots = cut_shots(frames, dur)
    sheets = make_sheets(yid, path, shots)
    rec = {"yid": yid, "series": series_of(yid), "src": path, "dur": round(dur, 3),
           "n_shots": len(shots), "sheets": sheets, "shots": shots}
    json.dump(rec, open(os.path.join(OUT, yid + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    long_ = sum(1 for s in shots if s["dur"] >= 2.33)
    dark = sum(1 for s in shots if s["dark"] >= 0.5)
    print("%-4s %-12s %6.1fs  %3d 镜头  ≥1小节 %3d  偏暗 %3d  sheets %d" %
          (rec["series"], yid, dur, len(shots), long_, dark, len(sheets)), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--jobs", type=int, default=1, help="并行处理的源数（单条 ffmpeg 解码吃不满 32 线程）")
    a = ap.parse_args()
    os.makedirs(SHEETS, exist_ok=True)
    paths = src_paths()
    ids = a.only or (SF + FGO)
    summary = []
    todo = [y for y in ids if y in paths]
    for y in ids:
        if y not in paths:
            print("缺 conform:", y)
    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        for r in ex.map(lambda y: run_one(y, paths[y]), todo):
            summary.append({k: r[k] for k in ("yid", "series", "dur", "n_shots", "sheets")})
    if not a.only:
        json.dump(summary, open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print("done:", len(summary), "sources")


if __name__ == "__main__":
    main()

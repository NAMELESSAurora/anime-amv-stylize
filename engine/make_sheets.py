# -*- coding: utf-8 -*-
"""
为素材生成紧凑接触表 (contact sheet)，供子代理目视审阅。

为什么要自己生成：
  直接把 4K 抽帧丢给子代理会把上下文撑爆（之前批次 3 就是这样失败的）。
  这里统一压到小尺寸、拼成单张 4x3 网格，一张图 = 12 个时间点。

输出:
  sheets2/<stem>__<id>.jpg   接触表（1600x900 以内）
  sheets2/index.json         每个源的 ffprobe 元数据 + 接触表路径

用法:
  python make_sheets.py --dir "D:/Fate_AMV/Fate_AMV_无字高清" --only 01_FGO
  python make_sheets.py --src "path/to/x.mp4"
"""
import os, sys, subprocess, json, re, hashlib

FFDIR = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin"
FF = os.path.join(FFDIR, "ffmpeg.exe")
FP = os.path.join(FFDIR, "ffprobe.exe")
OUT = r"D:\A\provant\sheets2"
EXTS = {".mkv", ".mp4", ".webm", ".mov", ".avi"}

COLS, ROWS = 4, 3
CELL_W = 480          # 每格宽（4 列 => 1920，再缩到 1600）
FONT = "C:/Windows/Fonts/arial.ttf"


def ytid(name):
    m = re.findall(r"\[([A-Za-z0-9_\-]{11})\]", name)
    return m[-1] if m else hashlib.md5(name.encode("utf-8")).hexdigest()[:11]


def stem_of(path):
    b = os.path.splitext(os.path.basename(path))[0]
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in b)[:60]
    return "%s__%s" % (keep, ytid(b))


def probe(path):
    p = subprocess.run([FP, "-v", "quiet", "-print_format", "json",
                        "-show_streams", "-show_format", path],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    try:
        return json.loads(p.stdout)
    except Exception:
        return {}


def meta_of(path):
    j = probe(path)
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in j.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not v:
        return None
    fr = v.get("r_frame_rate", "0/1")
    try:
        n, d = fr.split("/")
        fps = round(float(n) / float(d), 3) if float(d) else 0.0
    except Exception:
        fps = 0.0
    dur = j.get("format", {}).get("duration")
    return {
        "codec": v.get("codec_name"),
        "width": v.get("width"), "height": v.get("height"),
        "fps": fps, "pix_fmt": v.get("pix_fmt"),
        "duration": round(float(dur), 2) if dur else None,
        "has_audio": a is not None,
        "audio_codec": a.get("codec_name") if a else None,
    }


def make_sheet(src, m, outdir=OUT, cols=COLS, rows=ROWS):
    """均匀采样 cols*rows 帧，拼成一张接触表"""
    from PIL import Image, ImageDraw, ImageFont
    dur = m.get("duration") or 0
    if dur <= 0:
        return None
    n = cols * rows
    # 跳过首尾 4%，这两段通常是标题卡/黑场
    t0, t1 = dur * 0.04, dur * 0.96
    times = [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]

    cell_h = int(CELL_W * (m["height"] / float(m["width"])))
    W, H = CELL_W * cols, cell_h * rows
    sheet = Image.new("RGB", (W, H), (12, 12, 16))
    try:
        font = ImageFont.truetype(FONT, 17)
    except Exception:
        font = ImageFont.load_default()

    for i, t in enumerate(times):
        if i == n - 1:
            tmp = os.path.join(outdir, "_tmp_last.jpg")
            cmd = [FF, "-y", "-v", "error", "-sseof", "-%.3f" % max(0.05, dur - t),
                   "-i", src, "-frames:v", "1",
                   "-vf", "scale=%d:-2" % CELL_W, tmp]
        else:
            tmp = os.path.join(outdir, "_tmp_%02d.jpg" % i)
            cmd = [FF, "-y", "-v", "error", "-ss", "%.3f" % t, "-i", src,
                   "-frames:v", "1", "-vf", "scale=%d:-2" % CELL_W, tmp]
        subprocess.run(cmd, capture_output=True, timeout=300)
        if not os.path.exists(tmp):
            continue
        try:
            im = Image.open(tmp).convert("RGB")
            if im.size[1] != cell_h:
                im = im.resize((CELL_W, cell_h), Image.LANCZOS)
            cx, cy = (i % cols) * CELL_W, (i // cols) * cell_h
            sheet.paste(im, (cx, cy))
            d = ImageDraw.Draw(sheet)
            # 时间码底衬，保证任何画面上都能读
            label = "%d:%05.2f" % (int(t // 60), t % 60)
            bb = d.textbbox((0, 0), label, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            d.rectangle([cx + 4, cy + 4, cx + 12 + tw, cy + 10 + th], fill=(0, 0, 0))
            d.text((cx + 8, cy + 7), label, fill=(255, 235, 90), font=font)
        except Exception:
            pass
        finally:
            try: os.remove(tmp)
            except Exception: pass

    # 缩到最长边 1600，控制体积
    if max(sheet.size) > 1600:
        r = 1600.0 / max(sheet.size)
        sheet = sheet.resize((int(sheet.size[0] * r), int(sheet.size[1] * r)), Image.LANCZOS)
    dst = os.path.join(outdir, stem_of(src) + ".jpg")
    sheet.save(dst, "JPEG", quality=80, optimize=True)
    return dst


def main():
    argv = sys.argv[1:]
    only = None
    if "--only" in argv:
        i = argv.index("--only")
        only = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if "--dir" in argv:
        i = argv.index("--dir")
        root = argv[i + 1]
        srcs = []
        for dp, dn, fn in os.walk(root):
            if only and only.lower() not in dp.lower():
                continue
            for f in fn:
                if os.path.splitext(f)[1].lower() in EXTS:
                    srcs.append(os.path.join(dp, f))
    else:
        srcs = [a for a in argv if not a.startswith("--")]

    srcs = sorted(set(srcs))
    os.makedirs(OUT, exist_ok=True)
    index = []
    for n, s in enumerate(srcs, 1):
        m = meta_of(s)
        if not m:
            print("[%d/%d] SKIP (no video): %s" % (n, len(srcs), os.path.basename(s)[:50]))
            continue
        sheet = make_sheet(s, m)
        rec = {"src": s, "name": os.path.basename(s), "sheet": sheet, "meta": m}
        index.append(rec)
        print("[%d/%d] %s  %sx%s %sfps %ss  -> %s" % (
            n, len(srcs), m["codec"], m["width"], m["height"], m["fps"],
            m["duration"], os.path.basename(sheet) if sheet else "NO SHEET"))

    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print("\n%d source(s). index: %s" % (len(index), os.path.join(OUT, "index.json")))


if __name__ == "__main__":
    main()

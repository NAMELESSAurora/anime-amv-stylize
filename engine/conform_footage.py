# -*- coding: utf-8 -*-
"""
素材规范化 (conform)

为什么需要：
  AE 能直接读 .mp4/H.264，但读不了 .mkv，也读不了 AV1。
  另外 4K/60fps 的源在 AE 里实时预览极慢，而本片交付母版是 1920x1080/24fps。
  所以统一转成 ProRes 422 HQ 1080p24 作为剪辑中间格式。

注意：
  60fps 源是粉丝用 AI 插帧做的（动画原生 24fps）。这里 fps=24 会**丢帧**，
  也就是回到接近原始帧序列的状态——这正是我们想要的，比保留插帧更干净。
  如果某段素材需要保留插帧的运动感，用 --keep-fps。

用法:
  python conform_footage.py --list sources.txt
  python conform_footage.py "path/to/a.mkv" "path/to/b.mp4"
  python conform_footage.py --dir "D:/Fate_AMV/Fate_AMV_无字高清" --limit 3
  python conform_footage.py --size 2560x1440 --keep-fps --profile 2 --out D:/A/provant/media/conform_1440 --list x.txt
    (2026-10-17 起母版改为 2560×1440/60fps：高分辨率源按原帧率转 1440p；ProRes 422 标准档控制体积)
"""
import os, sys, subprocess, json, hashlib

FF = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
OUTDIR = r"D:\A\provant\media\conform"
EXTS = {".mkv", ".mp4", ".webm", ".mov", ".avi"}

TARGET_W, TARGET_H, TARGET_FPS = 1920, 1080, 24
PROFILE = 3   # ProRes 422 HQ；--profile 2 = 422 标准


def safe_name(p, maxlen=70):
    base = os.path.splitext(os.path.basename(p))[0]
    # 提取 YouTube ID（方括号里的 11 位），保证名字短且唯一
    import re
    m = re.findall(r"\[([A-Za-z0-9_\-]{11})\]", base)
    tag = m[-1] if m else hashlib.md5(p.encode("utf-8")).hexdigest()[:11]
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in base)[:maxlen]
    return "%s__%s" % (keep, tag)


def probe(path):
    p = subprocess.run(
        [FF.replace("ffmpeg.exe", "ffprobe.exe"), "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    try:
        return json.loads(p.stdout)
    except Exception:
        return {}


def conform(src, outdir=OUTDIR, keep_fps=False):
    if not os.path.exists(src):
        return {"src": src, "ok": False, "error": "file missing"}

    j = probe(src)
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        return {"src": src, "ok": False, "error": "no video stream"}

    codec = v.get("codec_name")
    w, h = v.get("width"), v.get("height")
    fr = v.get("r_frame_rate", "0/1")
    try:
        num, den = fr.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0

    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, safe_name(src) + ".mov")

    if os.path.exists(dst) and os.path.getsize(dst) > 1024 * 1024:
        return {"src": src, "ok": True, "out": dst, "cached": True,
                "codec": codec, "srcres": "%sx%s" % (w, h), "srcfps": round(fps, 3)}

    # 缩放：保持宽高比，统一到 1080p 高度；源宽高比若非 16:9 则补边
    vf = ("scale=%d:%d:force_original_aspect_ratio=decrease:flags=lanczos,setsar=1,"
          "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=black" % (TARGET_W, TARGET_H, TARGET_W, TARGET_H))
    if not keep_fps:
        vf += ",fps=%d" % TARGET_FPS

    cmd = [FF, "-y", "-v", "error", "-stats", "-i", src,
           "-vf", vf, "-c:v", "prores_ks", "-profile:v", str(PROFILE),
           "-pix_fmt", "yuv422p10le", "-an", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    ok = (r.returncode == 0 and os.path.exists(dst))
    return {
        "src": src, "out": dst, "ok": ok,
        "codec": codec, "srcres": "%sx%s" % (w, h), "srcfps": round(fps, 3),
        "kept_fps": keep_fps,
        "size_mb": round(os.path.getsize(dst) / 1048576.0, 1) if ok else None,
        "error": None if ok else (r.stderr or "")[-300:]
    }


def main():
    argv = sys.argv[1:]
    outdir = OUTDIR
    global TARGET_W, TARGET_H, PROFILE
    for flag in ("--size", "--profile"):
        if flag in argv:
            i = argv.index(flag)
            if flag == "--size":
                TARGET_W, TARGET_H = (int(v) for v in argv[i + 1].lower().split("x"))
            else:
                PROFILE = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2:]
    keep_fps = "--keep-fps" in argv
    argv = [a for a in argv if a != "--keep-fps"]
    limit = None
    if "--limit" in argv:
        i = argv.index("--limit")
        limit = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if "--out" in argv:
        i = argv.index("--out")
        outdir = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    srcs = []
    if "--list" in argv:
        i = argv.index("--list")
        with open(argv[i + 1], encoding="utf-8") as f:
            srcs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    elif "--dir" in argv:
        i = argv.index("--dir")
        root = argv[i + 1]
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if os.path.splitext(f)[1].lower() in EXTS:
                    srcs.append(os.path.join(dp, f))
    else:
        srcs = [a for a in argv if not a.startswith("--")]

    srcs = sorted(set(srcs))
    if limit:
        srcs = srcs[:limit]

    # conform() 会在"文件不存在"等分支提前 return，不建目录；
    # 那样最后的 manifest 写入会 FileNotFoundError。这里先建好。
    os.makedirs(outdir, exist_ok=True)

    print("%d source(s) -> %s\n" % (len(srcs), outdir))
    results = []
    for n, s in enumerate(srcs, 1):
        r = conform(s, outdir, keep_fps)
        results.append(r)
        flag = "CACHED" if r.get("cached") else ("OK" if r["ok"] else "FAIL")
        print("[%d/%d] %-7s %-34s %s %s fps -> %s MB" % (
            n, len(srcs), flag,
            os.path.basename(r["src"])[:34],
            r.get("srcres", "?"), r.get("srcfps", "?"),
            r.get("size_mb", r.get("error", ""))))

    with open(os.path.join(outdir, "conform_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    okc = sum(1 for r in results if r["ok"])
    print("\n%d/%d conformed. manifest: %s" % (okc, len(results),
          os.path.join(outdir, "conform_manifest.json")))


if __name__ == "__main__":
    main()

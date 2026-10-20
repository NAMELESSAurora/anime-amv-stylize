import subprocess, sys, json, numpy as np
sys.path.insert(0, "scripts")
from cards import src_path
FF = r"E:\NAMELESS\tools\ffmpeg-9.0.1\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
P = {r["code"]: r for r in json.load(open("data/shot_pool_v2.json", encoding="utf-8"))}
for c in sys.argv[1:]:
    r = P[c]
    p = subprocess.run([FF, "-v", "error", "-ss", "%.3f" % r["in"], "-t", "%.3f" % r["dur"], "-i", src_path(r["yid"]), "-vf", "scale=160:90",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    fr = np.frombuffer(p.stdout, np.uint8); n = len(fr) // 14400
    m = fr[:n * 14400].reshape(n, -1).mean(1)
    fps = n / r["dur"]
    j = np.nonzero(np.abs(np.diff(m * 0.78)) >= 51)[0]
    print(c, "%.3f-%.3f" % (r["in"], r["out"]), "fps %.1f" % fps, "jumps@", [round(r["in"] + (k + 1) / fps, 2) for k in j])
    print("   ", " ".join("%d" % v for v in m[::max(1, n // 40)]))

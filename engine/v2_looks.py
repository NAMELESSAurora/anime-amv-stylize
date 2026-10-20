# -*- coding: utf-8 -*-
"""
v2 全局风格层（用户 10-18："全局都要进行风格化处理…不能只放原片"，第三种问过答"两种都用"）：

  R  Raf0x00 战术 HUD（战斗 / 宝具）：原画面保留色彩、压暗，阴影走冷色半调网点 + 细颗粒（每 2 帧换）；
     叠加 HUD：同心轨道环 / 90-120-180° 弧段 / 刻度环（每拍步进 15°，6 帧缓动到位）、准星角标锁定人物或脸、
     等宽坐标小字、点阵网格（≤¼ 画面，Y≤110，放在主体对侧）、歌曲 RMS 波形条 + 原始波形线、重拍上波形撕裂。
  C  causasui 档案（人物 / 抒情）：
     暗态 = 人物保留原色、背景压暗去饱和，白色星座线挂在人物轮廓上（节点 back-out 弹出、边依次画出）+ 大写标签 + 坐标；
     亮态 = 纸灰底（Y≤150）+ 人物阈值灰阶 + 裁切框（脸 / 细节特写，锈色标题条 + 引线，按拍弹出）+ 锈色像素块生长
            （≤10% 画面，离脸 ≥8×R）。
  歌词：HUD 段解码 / 打字机，档案段进标题条 / 衬线大字；文本按 i 从用户对轴包读取（不入库）。

锚点：scripts/anime_det.py（deepghs YOLOv8 脸 / 人物检测）+ models/anime-seg/isnetis.onnx（人物抠像）。
本文件先做风格帧（--stills），逐帧序列渲染在同一套函数上加时间轴。
"""
import json, math, os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
import style_frames as SF
from style_frames import W, H, R, font, gray
from cards import halftone_c, eo, bo, cl

ROOT = SF.ROOT
QA = os.path.join(ROOT, "ae", "qa", "v2style")
LYR = os.path.join(ROOT, "research", "lyrics_user", "ja_en", "PROVANT_逐句时间数据.json")
F_JP = os.path.join(SF.FONTS, "YuGothB.ttc")          # 日文黑体（HUD / 标签）
F_JPS = SF.F_SERIF_CJK                                # 明朝风（档案段大字）
F_EN = SF.F_BAHN
F_WIDE, F_MONO = SF.F_WIDE, SF.F_MONO

RUST = (196, 92, 44)
PAPER_G = (150, 147, 140)        # 纸灰 Y≈147（包规则 ≤150）
INK = (14, 13, 12)
BLUEBLK = (22, 30, 46)
COLDG = (92, 98, 104)
LIGHT = (176, 174, 168)
HUD = (214, 228, 234)            # HUD 主线：冷白
HUD_D = (120, 142, 152)          # HUD 次线
WAVE = (114, 118, 147)           # Raf0x00 的灰蓝 #727693


# ---------------------------------------------------------------- 强调色：跟画面主色走
# 用户 10-18："取景框不能只是橘色的，要根据画面主色进行变幻"。
# 取法：缩到 160×90，按 饱和度^1.5 × 亮度 加权做 36 档色相直方图（环形平滑），主峰 = 主色相，
# 次峰（≥ 主峰 25%、相距 ≥ 60°）= 副色相（没有就用主色相 +35° 的邻近色）。
#   main  饱和强调色（标题条 / 像素块 / 角标实心块），S 0.55–0.9，V 0.9
#   hi    浅色调（线 / 取景框角标 / 引线，压在同色画面上也看得见）
#   dk    深色调（纸灰底上的字 / 细线）
#   txt   压在 main 上的字色（按 main 的亮度选墨色或米白）
# 画面几乎无彩色时退回锈色。逐帧用时用 AccentTrack 做时间平滑（色相走最短弧）。
def _hsv2rgb(h, s, v):
    c = np.uint8([[[int(h / 2) % 180, int(np.clip(s, 0, 1) * 255), int(np.clip(v, 0, 1) * 255)]]])
    return tuple(int(x) for x in cv2.cvtColor(c, cv2.COLOR_HSV2RGB)[0, 0])


def hue_peaks(rgb, mask=None):
    sm = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0] * 2.0, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    w = s ** 1.5 * np.clip((v - 0.15) / 0.5, 0, 1)
    if mask is not None:
        w = w * (0.35 + 0.65 * cv2.resize(mask.astype(np.float32), (160, 90)).reshape(-1))
    hist = np.bincount((h // 10).astype(int) % 36, weights=w, minlength=36)
    k = np.array([1, 2, 3, 2, 1], np.float32)
    hs = sum(np.roll(hist, i - 2) * k[i] for i in range(5)) / k.sum()
    colorful = float(w.mean())
    p1 = int(np.argmax(hs))
    def center(p):
        d = (h - (p * 10 + 5) + 180) % 360 - 180
        b = np.abs(d) < 25
        if w[b].sum() <= 0:
            return p * 10 + 5.0, 0.6
        return float((p * 10 + 5 + np.average(d[b], weights=w[b])) % 360), float(np.average(s[b], weights=w[b]))
    h1, s1 = center(p1)
    far = np.array([abs(((i * 10 + 5) - h1 + 180) % 360 - 180) >= 60 for i in range(36)])
    p2 = int(np.argmax(np.where(far, hs, -1)))
    h2 = center(p2)[0] if hs[p2] >= 0.25 * hs[p1] else (h1 + 35) % 360
    return h1, s1, h2, colorful


def accent_from(h1, s1, h2, colorful):
    if colorful < 0.04:                       # 几乎无彩色（黑白 / 灰夜景）：冷钢蓝（用户："少用橙色配色"）
        h1, s1, h2 = 210.0, 0.55, 350.0
    S = float(np.clip(s1 * 1.15, 0.55, 0.9))
    main = _hsv2rgb(h1, S, 0.9)
    hi = _hsv2rgb(h1, S * 0.42, 1.0)
    dk = _hsv2rgb(h1, min(1.0, S * 1.05), 0.42)
    sec = _hsv2rgb(h2, 0.5, 0.95)
    y = 0.299 * main[0] + 0.587 * main[1] + 0.114 * main[2]
    txt = INK if y > 150 else (246, 240, 232)
    return dict(main=main, hi=hi, dk=dk, sec=sec, txt=txt, hue=h1, hue2=h2)


def accent_of(rgb, mask=None):
    return accent_from(*hue_peaks(rgb, mask))


class AccentTrack:
    """逐帧强调色：色相做指数平滑（最短弧，τ≈20 帧），换镜头时 reset 后 10 帧缓到新色。"""
    def __init__(self, tau=20.0):
        self.a = 1.0 / tau; self.st = None

    def __call__(self, rgb, mask=None):
        h1, s1, h2, c = hue_peaks(rgb, mask)
        if self.st is None:
            self.st = [h1, s1, h2, c]
        else:
            for i, (x, circ) in enumerate(((h1, 1), (s1, 0), (h2, 1), (c, 0))):
                if circ:
                    d = (x - self.st[i] + 180) % 360 - 180
                    self.st[i] = (self.st[i] + self.a * d) % 360
                else:
                    self.st[i] += self.a * (x - self.st[i])
        return accent_from(*self.st)


def mix_acc(a, b, k):
    """两个强调色之间插值（转场用）：色相走最短弧。"""
    k = cl(k)
    out = {}
    for key in ("main", "hi", "dk", "sec"):
        out[key] = tuple(int(round(a[key][i] * (1 - k) + b[key][i] * k)) for i in range(3))
    out["txt"] = a["txt"] if k < 0.5 else b["txt"]
    out["hue"] = (a["hue"] + k * ((b["hue"] - a["hue"] + 180) % 360 - 180)) % 360
    out["hue2"] = b["hue2"]
    return out


RUST_ACC = accent_from(20.0, 0.75, 200.0, 1.0)


# ---------------------------------------------------------------- 锚点：检测 / 抠像
_SEG = None


def seg_mask(rgb):
    """人物抠像 0..1（isnetis，1024 输入）。"""
    global _SEG
    import onnxruntime as ort
    if _SEG is None:
        so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("SEG_THREADS", "4"))
        prov = ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in ort.get_available_providers():      # .venv_gpu（onnxruntime-directml）：跑在显卡上
            so.enable_mem_pattern = False; so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
        _SEG = ort.InferenceSession(os.path.join(ROOT, "models", "anime-seg", "isnetis.onnx"), so, providers=prov)
    S = 1024
    h, w = rgb.shape[:2]; r = S / max(h, w); nh, nw = int(h * r), int(w * r)
    x = np.zeros((S, S, 3), np.float32); y0, x0 = (S - nh) // 2, (S - nw) // 2
    x[y0:y0 + nh, x0:x0 + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA) / 255.0
    m = _SEG.run(None, {_SEG.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    return np.clip(cv2.resize(m[y0:y0 + nh, x0:x0 + nw], (w, h)), 0, 1)


def face_mask(faces, pad=8):
    m = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1, _ in faces:
        p = pad * R
        cv2.ellipse(m, (int((x0 + x1) / 2), int((y0 + y1) / 2)), (int((x1 - x0) / 2 + p), int((y1 - y0) / 2 + p)), 0, 0, 360, 255, -1)
    return m


# ---------------------------------------------------------------- 文字
def is_cjk(ch):
    o = ord(ch)
    return o >= 0x2E80 or 0x3000 <= o <= 0x30FF or o == 0xFF1F


def runs(text):
    out = []
    for ch in text:
        c = is_cjk(ch) if ch != " " else (out[-1][0] if out else False)
        if out and out[-1][0] == c:
            out[-1][1] += ch
        else:
            out.append([c, ch])
    return out


def mixed(text, size, fill, jp=F_JP, en=F_EN, en_scale=1.0, track=0.0, shown=None, scramble=None):
    """日英混排单行 → RGBA。shown=只显示前 n 个字；scramble={索引: 替代字} 用于解码效果。"""
    fj, fe = font(jp, size), font(en, size * en_scale)
    fm = font(F_MONO, size * 0.8)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    asc = max(fj.getmetrics()[0], fe.getmetrics()[0]); desc = max(fj.getmetrics()[1], fe.getmetrics()[1])
    wsum = 0
    for c, s in runs(text):
        f = fj if c else fe
        wsum += sum(probe.textlength(ch, font=f) + track for ch in s)
    img = Image.new("RGBA", (int(wsum + size), asc + desc + 4), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    x, k = 0.0, 0
    for c, s in runs(text):
        f = fj if c else fe
        for ch in s:
            adv = probe.textlength(ch, font=f) + track
            if shown is None or k < shown:
                if scramble and k in scramble:
                    d.text((x, asc), scramble[k], font=fm, fill=fill, anchor="ls")
                else:
                    d.text((x, asc), ch, font=f, fill=fill, anchor="ls")
            x += adv; k += 1
    return img


def decode(text, f, f0, cps=90.0, settle=6, seed=0):
    """解码 / 打字机：90 字/秒出字，每字先乱码 settle 帧再定字。返回 (shown, scramble)。"""
    n = len(text)
    per = 60.0 / cps
    shown = max(0, min(n, int((f - f0) / per) + 1)) if f >= f0 else 0
    rng = np.random.default_rng(seed * 1000 + int(f) // 2)
    glyph = "01#%/<>=+:;"
    scr = {}
    for i in range(shown):
        if f - (f0 + i * per) < settle and text[i] != " ":
            scr[i] = glyph[int(rng.integers(len(glyph)))]
    return shown, scr


def lyric(i):
    return json.load(open(LYR, encoding="utf-8"))["rows"][i]


# ---------------------------------------------------------------- 音频
_RMS = None


def rms60():
    """歌曲 RMS（60 fps 一帧一值，0..1），缓存在 data/rms60.npy。"""
    global _RMS
    if _RMS is None:
        p = os.path.join(ROOT, "data", "rms60.npy")
        if not os.path.exists(p):
            import librosa, glob
            wav = [q for q in glob.glob(r"D:\备份\*.wav") if "PROVANT" in os.path.basename(q)][0]
            y, sr = librosa.load(wav, sr=48000, mono=True)
            hop = 800
            r = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
            np.save(p, r / (np.percentile(r, 99.5) + 1e-9))
            np.save(os.path.join(ROOT, "data", "wav48_mono.npy"), y.astype(np.float16))
        _RMS = np.load(p)
    return _RMS


_WAV = None


def wav_at(t, dur=1 / 30.0, n=400):
    global _WAV
    if _WAV is None:
        rms60()
        _WAV = np.load(os.path.join(ROOT, "data", "wav48_mono.npy")).astype(np.float32)
    a = int(t * 48000); b = a + int(dur * 48000)
    seg = _WAV[max(0, a):max(0, b)]
    if len(seg) < 2:
        return np.zeros(n)
    return np.interp(np.linspace(0, len(seg) - 1, n), np.arange(len(seg)), seg)


# ---------------------------------------------------------------- R · 战术 HUD
def r_base(src, f, strength=1.0):
    """原色压暗 + 阴影冷色半调 + 细颗粒（hold 2）。有显卡时走 gpu.r_base（同算法的着色器）。"""
    import gpu
    if gpu.use():
        return gpu.r_base(src, f, strength)
    g = gray(src)
    gf = g.astype(np.float32)[..., None] / 255.0
    dot = (halftone_c(g, 7 * R, 15) == 0)[..., None]
    w = np.clip((0.55 - gf) / 0.35, 0, 1) * strength
    out = src.astype(np.float32) * (1 - 0.22 * strength)
    out = out * (1 - 0.28 * w) + np.array([18, 38, 50], np.float32) * 0.28 * w        # 阴影偏冷
    out = np.where(dot, out * (1 - 0.7 * w) + np.array([6, 12, 18], np.float32) * 0.7 * w, out)
    rng = np.random.default_rng(int(f) // 2)
    grain = rng.normal(0, 5.0 * strength, (H // 2, W // 2)).astype(np.float32)
    out += cv2.resize(grain, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def wave_tear(img, k, bands=((0.40, 0.03), (0.53, 0.015), (0.62, 0.02)), amp=40, tint=RUST):
    """重拍上的波形撕裂：3 条带水平错位 + 锈色残影；k 0..1 进出（3 帧出、8 帧回由调用方给）。"""
    env = math.sin(math.pi * cl(k)) ** 2
    if env <= 0.01:
        return img
    out = img.copy(); A = amp * R * env
    for j, (cy, hh) in enumerate(bands):
        y0, y1 = int(cy * H), int((cy + hh) * H)
        sh = int(A * (1 if j % 2 == 0 else -0.6))
        band = np.roll(img[y0:y1], sh, axis=1).astype(np.float32)
        ghost = np.roll(img[y0:y1], sh + int(6 * R), axis=1).astype(np.float32) * (np.array(tint, np.float32) / 255)
        out[y0:y1] = np.clip(band * (1 - 0.35 * env) + ghost * 0.35 * env * 1.6, 0, 255).astype(np.uint8)
    return out


class HudR:
    def __init__(self, target, faces=(), tags=(), head="", seed=0, acc=None):
        """target = (x0, y0, x1, y1) 锁定框（人物或脸）。acc = 强调色（逐帧可改 self.acc）。"""
        self.acc = acc or RUST_ACC
        self.t = target; self.faces = list(faces); self.tags = list(tags); self.head = head; self.seed = seed
        x0, y0, x1, y1 = target
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.fm = font(F_MONO, 11 * R); self.fs = font(F_MONO, 9 * R); self.fw = font(F_WIDE, 13 * R)
        self.fh = font(F_WIDE, 22 * R)

    def set_target(self, box):
        self.t = tuple(box)
        self.cx, self.cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def rings(self, d, f, beat_k, beat_f):
        cx, cy = self.cx, self.cy
        step = 15 * (beat_k - 1 + eo(beat_f / 6.0))               # 每拍步进 15°，6 帧缓动到位
        spec = ((150, 2, 0, 180, 1), (205, 1, 30, 150, -1), (250, 4, 200, 290, 1), (300, 1, 0, 360, 0),
                (360, 1, 100, 220, -1), (430, 2, 250, 340, 1), (520, 1, 0, 360, 0))
        for r, w_, a0, a1, sgn in spec:
            rr = r * R
            rot = step * sgn
            col = (self.acc["main"] + (235,) if w_ >= 4 else HUD + (220,)) if w_ > 1 else HUD_D + (170,)
            if a1 - a0 >= 360:
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=col, width=max(1, int(w_ * R)))
            else:
                d.arc([cx - rr, cy - rr, cx + rr, cy + rr], a0 + rot, a1 + rot, fill=col, width=max(1, int(w_ * R)))
        for k in range(120):                                       # 刻度环
            a = math.radians(k * 3 + step * 0.5)
            r0 = 455 * R; r1 = r0 + (16 if k % 10 == 0 else (8 if k % 2 == 0 else 4)) * R
            d.line([cx + r0 * math.cos(a), cy + r0 * math.sin(a), cx + r1 * math.cos(a), cy + r1 * math.sin(a)],
                   fill=HUD_D + (150,), width=max(1, int(R * 0.8)))
        # 沿轨道排的小字（每拍跟着步进）
        txt = "MANA OUTPUT // ORBIT %03d // PROVANT" % (beat_k % 1000)
        rr = 395 * R
        for i, ch in enumerate(txt):
            a = math.radians(-120 + step + i * 2.4)
            d.text((cx + rr * math.cos(a), cy + rr * math.sin(a)), ch, font=self.fs, fill=HUD_D + (200,), anchor="mm")

    def lock(self, d, f, lock_f=14):
        x0, y0, x1, y1 = self.t
        k = eo(f / float(lock_f))
        pad = (1 - k) * 160 * R
        L = 48 * R; w_ = max(2, int(2 * R))
        A = self.acc
        col = A["hi"] + (255,) if f >= lock_f else (A["hi"] if int(f / 2) % 2 == 0 else A["main"]) + (255,)
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            x = (x0 if sx < 0 else x1) + sx * pad; y = (y0 if sy < 0 else y1) + sy * pad
            d.line([x, y, x - sx * L, y], fill=col, width=w_)
            d.line([x, y, x, y - sy * L], fill=col, width=w_)
        # 准星十字线：从框边伸到画面边
        gp = 40 * R; ext = eo((f - 2) / 10.0)
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if sx:
                xa = (x1 if sx > 0 else x0) + sx * gp; xb = xa + sx * W * ext
                d.line([xa, self.cy, xb, self.cy], fill=HUD_D + (110,), width=max(1, int(R)))
            else:
                ya = (y1 if sy > 0 else y0) + sy * gp; yb = ya + sy * H * ext
                d.line([self.cx, ya, self.cx, yb], fill=HUD_D + (110,), width=max(1, int(R)))
        if f >= lock_f:
            d.text((x0, y1 + 10 * R), "LOCK  %.3f , %.3f" % (self.cx / W, self.cy / H), font=self.fm, fill=HUD + (255,))
            bw = d.textlength("[ ENGAGED ]", font=self.fm)
            d.rectangle([x1 - bw - 8 * R, y1 + 8 * R, x1, y1 + 26 * R], fill=A["main"] + (255,))
            d.text((x1 - bw - 4 * R, y1 + 10 * R), "[ ENGAGED ]", font=self.fm, fill=A["txt"] + (255,))

    def dots(self, d):
        # 主体对侧 ¼ 画面，点距 24R，点 Y≈90
        qx = 0 if self.cx > W / 2 else W / 2; qy = 0 if self.cy > H / 2 else H / 2
        s = 24 * R
        for yy in np.arange(qy + s, qy + H / 2 - s / 2, s):
            for xx in np.arange(qx + s, qx + W / 2 - s / 2, s):
                d.rectangle([xx, yy, xx + max(1, R * 1.2), yy + max(1, R * 1.2)], fill=(92, 96, 100, 255))

    def wave(self, d, t):
        r = rms60(); i = int(t * 60)
        x0, yb = 60 * R, H - 70 * R
        n = 120
        for k in range(n):
            v = r[max(0, i - n + k)] if i - n + k < len(r) else 0
            h = 4 * R + 70 * R * min(1.0, v)
            x = x0 + k * 4 * R
            d.rectangle([x, yb - h, x + 2 * R, yb], fill=HUD_D + (200,))
        d.text((x0, yb + 8 * R), "RMS  %.3f   T+%07.3f" % (r[min(i, len(r) - 1)], t), font=self.fs, fill=HUD_D + (220,))
        w = wav_at(t)
        pts = [(W * 0.52 + k * (W * 0.44) / len(w), H - 110 * R - w[k] * 120 * R) for k in range(len(w))]
        d.line(pts, fill=WAVE + (230,), width=max(1, int(1.5 * R)))
        d.line([W * 0.52, H - 110 * R, W * 0.96, H - 110 * R], fill=WAVE + (120,), width=1)

    def panel(self, d, f):
        # 左上：参数栏（数字按 3 帧滚动）
        x, y = 60 * R, 60 * R
        if self.head:
            d.text((x, y), self.head, font=self.fh, fill=HUD + (255,))
            d.rectangle([x, y + 30 * R, x + 90 * R, y + 33 * R], fill=self.acc["main"] + (255,))
            y += 44 * R
        for i, tg in enumerate(self.tags):
            d.text((x, y + i * 20 * R), tg, font=self.fm, fill=HUD + (230,))
        y += (len(self.tags) + 1) * 20 * R
        rng = np.random.default_rng(self.seed + int(f) // 3)
        for i in range(8):
            v = rng.random() * 360
            d.text((x, y + i * 16 * R), "%02d  %06.2f  %s" % (i, v, "▮" * int(rng.integers(1, 6))), font=self.fs,
                   fill=HUD_D + (190,))

    def overlay(self, f, t, beat_k, beat_f):
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        self.dots(d)
        self.rings(d, f, beat_k, beat_f)
        self.lock(d, f)
        self.wave(d, t)
        self.panel(d, f)
        return lay


def lyric_hud(lay, text, f, f0, xy, size=40, seed=0, acc=None):
    """HUD 段歌词：黑底标签条 + 解码出字；左侧强调色竖条。"""
    acc = acc or RUST_ACC
    shown, scr = decode(text, f, f0, seed=seed)
    img = mixed(text, size * R, (240, 240, 236, 255), shown=shown, scramble=scr)
    full = mixed(text, size * R, (0, 0, 0, 255))
    x, y = xy
    pad = 14 * R
    d = ImageDraw.Draw(lay)
    d.rectangle([x - pad, y - pad, x + full.width + pad * 0.3, y + full.height + pad * 0.6], fill=(8, 10, 12, 215))
    d.rectangle([x - pad, y - pad, x - pad + 6 * R, y + full.height + pad * 0.6], fill=acc["main"] + (255,))
    lay.alpha_composite(img, (int(x), int(y)))
    fs = font(F_MONO, 9 * R)
    d.text((x, y + full.height + pad * 0.9), "LYR %02d  //  DECODE %3d%%" % (seed, 100 * shown / max(1, len(text))),
           font=fs, fill=HUD_D + (220,))
    return lay


# ---------------------------------------------------------------- P · hectoroz 版画（高潮用）
# 原帖 https://x.com/hectoroz_/status/2111462081442832418：posterize + 阈值，中间调删掉、暗部压死、高光烧成金；
# 背景是阈值颗粒（版画 / 热成像），姿势每 2 帧重画。我们的改编：色阶按画面分位数切（输出平均亮度稳定，
# 和前后彩色镜头差 ≤ 约 15），主色 = 本镜头强调色（不是固定大红），不做亮暗逐帧交替（光敏规则），
# 颗粒每 2 帧换一次；"姿势沸腾"由调用方用 12 fps 步进取帧 + boil() 做。
GOLD_H = (240, 222, 128)
CREAM_H = (240, 236, 206)


def poster_h(src, f, acc, seed=0, mode="full"):
    """mode="tension"：换气段用的压暗版（主色降成深色调、只留金色边线、无米白），比 full 暗约 20，爆点时跳回 full。"""
    sm = cv2.resize(src, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    sm = cv2.bilateralFilter(sm, 7, 40, 7)
    g = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rng = np.random.default_rng(seed * 100 + int(f) // 2)
    g += cv2.resize(rng.normal(0, 9, (H // 8, W // 8)).astype(np.float32), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    q = np.quantile(g, [0.38, 0.66, 0.86, 0.965])
    deep = np.array(acc["dk"], np.float32); deep *= min(1.0, 38.0 / max(1.0, 0.299 * deep[0] + 0.587 * deep[1] + 0.114 * deep[2]))
    main = np.array(acc["main"], np.float32); main *= min(1.0, 100.0 / max(1.0, 0.299 * main[0] + 0.587 * main[1] + 0.114 * main[2]))
    out = np.empty(g.shape + (3,), np.float32); out[:] = (10, 4, 9)
    e = cv2.Canny(cv2.GaussianBlur(g.astype(np.uint8), (0, 0), 1.0), 50, 130) > 0
    if mode == "tension":
        out[g >= q[0]] = deep * 0.55; out[g >= q[1]] = deep; out[g >= q[2]] = main * 0.7
        out[e & (g >= q[1])] = GOLD_H
    else:
        out[g >= q[0]] = deep; out[g >= q[1]] = main; out[g >= q[2]] = GOLD_H; out[g >= q[3]] = CREAM_H
        out[e & (g >= q[1])] = GOLD_H
    return cv2.resize(out.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)


def boil(img, f, amp=1.6, scale=48, seed=0):
    """线条沸腾：低频位移场，每 2 帧换（hold 2），定格画面也"活"着；镜头本身不动。"""
    rng = np.random.default_rng(1000 + seed * 7 + int(f) // 2)
    s = int(scale * R); a = amp * R
    fld = lambda: cv2.resize(rng.standard_normal((H // s + 2, W // s + 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * a
    dx, dy = fld(), fld()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- C · causasui 档案
def nodes_from_mask(m, faces, n=8, seed=0):
    """星座节点：人物轮廓上取均匀点 + 脸中心；返回 [(x, y)]。"""
    b = (m > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = []
    if cs:
        c = max(cs, key=cv2.contourArea)[:, 0, :]
        # 只取不贴画面边的轮廓点
        ok = (c[:, 0] > 30 * R) & (c[:, 0] < W - 30 * R) & (c[:, 1] > 30 * R) & (c[:, 1] < H - 30 * R)
        c = c[ok]
        if len(c):
            idx = np.linspace(0, len(c) - 1, n + 1)[:-1].astype(int)
            pts = [tuple(map(int, c[i])) for i in idx]
            # 轮廓点往里收一点，线挂在人物身上而不是背景
            M = cv2.moments(b); gx, gy = M["m10"] / max(M["m00"], 1), M["m01"] / max(M["m00"], 1)
            pts = [(int(x + (gx - x) * 0.12), int(y + (gy - y) * 0.12)) for x, y in pts]
    for x0, y0, x1, y1, _ in faces:
        pts.append((int((x0 + x1) / 2), int(y0 + (y1 - y0) * 0.78)))       # 下巴附近，不压眼睛
    return pts


def edges_of(pts, k=2):
    """每个节点连最近的 k 个 + 顺序连，去重。"""
    E = set()
    P = np.array(pts, float)
    for i in range(len(pts)):
        d = np.hypot(*(P - P[i]).T); order = np.argsort(d)[1:k + 1]
        for j in order:
            E.add((min(i, j), max(i, j)))
        if i + 1 < len(pts):
            E.add((i, i + 1))
    return sorted(E)


def c_dark(src, m, f, nodes, labels, seed=0, acc=None):
    """暗态：主角保留原色；其余画面压成冷暗的印刷灰阶（45° 网点 + 颗粒，hold 2），
    白星座线挂在主角身上（边依次画出），节点带强调色芯，前两个标签做成强调色标签块；四角裁切标记。"""
    acc = acc or RUST_ACC
    import gpu
    tint = np.array(acc["dk"], np.float32); tint = 0.7 + 0.3 * tint / max(tint.max(), 1)
    if gpu.use():                                          # 显卡：印刷灰阶 + 网点 + 颗粒 + 抠像混合
        mh = cv2.GaussianBlur(cv2.resize(m.astype(np.float32), (W // 2, H // 2), interpolation=cv2.INTER_AREA), (0, 0), R)
        out = gpu.c_dark_base(src, (np.clip(mh, 0, 1) * 255).astype(np.uint8), tint, seed * 100 + int(f) // 2).astype(np.float32)
    else:
        g = gray(src)
        gf = g.astype(np.float32)[..., None] / 255.0
        tone = gf * 150.0 * tint
        dots = (halftone_c(g, 7 * R, 45) == 0)[..., None]
        bg = np.where(dots, tone * 0.35, tone)
        rng = np.random.default_rng(seed * 100 + int(f) // 2)
        bg = bg + cv2.resize(rng.normal(0, 4.0, (H // 2, W // 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        mm = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2 * R)[..., None]
        out = src.astype(np.float32) * mm + bg * (1 - mm)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    ft = font(F_WIDE, 13 * R); fm = font(F_MONO, 10 * R)
    E = edges_of(nodes)
    for (i, j) in E:                                     # 边依次画出（每条 8 帧）
        p = eo((f - 4 - 2 * max(i, j)) / 8.0)
        if p <= 0:
            continue
        a, b = np.array(nodes[i], float), np.array(nodes[j], float); c = a + (b - a) * p
        d.line([tuple(a), tuple(c)], fill=(242, 242, 238, 235), width=max(1, int(1.5 * R)))
    for i, ((x, y), lb) in enumerate(zip(nodes, labels)):
        k = (f - 2 * i) / 5.0
        if k <= 0:
            continue
        r = 4.5 * R * bo(k)
        d.ellipse([x - r, y - r, x + r, y + r], outline=(248, 248, 244, 255), width=max(1, int(1.5 * R)))
        q = r * 0.45
        d.rectangle([x - q, y - q, x + q, y + q], fill=acc["hi"] + (255,))
        tx, ty = x + 10 * R, y - 20 * R
        if i < 2:                                        # 强调色标签块
            tw = d.textlength(lb, font=ft)
            d.rectangle([tx - 4 * R, ty - 2 * R, tx + tw + 4 * R, ty + 16 * R], fill=acc["main"] + (255,))
            d.text((tx, ty), lb, font=ft, fill=acc["txt"] + (255,))
        else:
            d.text((tx, ty), lb, font=ft, fill=(248, 248, 244, 255))
    L_ = 36 * R; mg = 40 * R; w_ = max(1, int(1.5 * R))   # 四角裁切标记
    for (x, y, sx, sy) in ((mg, mg, 1, 1), (W - mg, mg, -1, 1), (mg, H - mg, 1, -1), (W - mg, H - mg, -1, -1)):
        d.line([x, y, x + sx * L_, y], fill=(220, 220, 214, 200), width=w_)
        d.line([x, y, x, y + sy * L_], fill=(220, 220, 214, 200), width=w_)
    return np.clip(out, 0, 255).astype(np.uint8), lay


def palette_c(src, m, acc=None):
    """亮态人物：灰阶 4 阶映射到 墨 / 蓝黑 / 冷灰 / 浅灰，中间调加 45° 网点；高光边缘烧锈色。"""
    g = cv2.GaussianBlur(gray(src), (0, 0), 1.0)
    gn = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX, mask=(m > 0.5).astype(np.uint8))
    out = np.zeros((H, W, 3), np.uint8)
    for th, c in ((0, INK), (55, BLUEBLK), (100, COLDG), (165, LIGHT)):
        out[gn >= th] = c
    ht = halftone_c(gn, 6 * R, 45)
    mid = (gn >= 100) & (gn < 165) & (ht == 0)
    out[mid] = BLUEBLK
    e = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.2 * R), 60, 140)
    out[(cv2.dilate(e, None) > 0) & (gn > 165)] = (acc or RUST_ACC)["main"]
    return out


def pixel_growth(lay, m, fmask, k, seeds, cell=12, cap=0.10, col=RUST):
    """锈色像素块：从人物轮廓外侧种子点生长（Bayer 顺序，无随机闪），避开脸（已外扩），面积 ≤ cap。"""
    c = int(cell * R)
    gy, gx = np.mgrid[0:H // c, 0:W // c]
    BAY = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
    edge = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((int(40 * R),) * 2, np.uint8))
    eg = edge[::c, ::c][:gy.shape[0], :gx.shape[1]] > 0
    fm = fmask[::c, ::c][:gy.shape[0], :gx.shape[1]] > 0
    best = np.full(gy.shape, 9.0)
    for ox, oy, rad in seeds:
        dist = np.hypot(gx * c - ox, gy * c - oy) / (rad * R) + BAY[gy % 4, gx % 4] * 0.4
        okk = (dist < 1.0) & (eg | (dist < 0.45)) & ~fm
        best = np.where(okk, np.minimum(best, dist), best)
    on = best < k
    lim = int(cap * (W // c) * (H // c))
    if on.sum() > lim:
        th = np.sort(best[on])[lim]; on &= best < th
    d = ImageDraw.Draw(lay)
    ys, xs = np.nonzero(on)
    for y, x in zip(ys, xs):
        d.rectangle([x * c, y * c, x * c + c - max(1, int(R)), y * c + c - max(1, int(R))], fill=tuple(col) + (255,))
    return float(on.sum()) / ((W // c) * (H // c))


def crop_box(lay, src, rect, dst, size, title, k, ink=INK, acc=None):
    acc = acc or RUST_ACC
    """裁切框：源区域灰度高反差特写 → 1.38→0.972→1.01→1 拍贴；锈色标题条 + 引线。"""
    x, y, w, h = [int(v) for v in rect]
    s = bo(k, 2.2) if k < 1 else 1.0
    if s <= 0.02:
        return
    crop = cv2.cvtColor(src[max(0, y):y + h, max(0, x):x + w], cv2.COLOR_RGB2GRAY)
    crop = cv2.convertScaleAbs(crop, alpha=1.35, beta=-30)
    bw, bh = int(size[0] * s), int(size[1] * s)
    X, Y = int(dst[0] + (size[0] - bw) / 2), int(dst[1] + (size[1] - bh) / 2)
    im = Image.fromarray(cv2.resize(crop, (max(2, bw), max(2, bh)), interpolation=cv2.INTER_AREA)).convert("RGBA")
    d = ImageDraw.Draw(lay)
    d.line([x + w / 2, y + h / 2, X, Y + bh / 2], fill=acc["main"] + (255,), width=max(1, int(R)))   # 引线先在底下
    lay.alpha_composite(im, (X, Y))
    d.rectangle([X, Y, X + bw, Y + bh], outline=ink + (255,), width=max(1, int(R)))
    bar = int(18 * R)
    d.rectangle([X, Y - bar, X + bw, Y], fill=acc["main"] + (255,))
    d.text((X + 6 * R, Y - bar + 3 * R), title, font=font(F_WIDE, 10 * R), fill=acc["txt"] + (255,))
    d.rectangle([x, y, x + w, y + h], outline=acc["main"] + (255,), width=max(1, int(R)))       # 源区域标记


def c_bright(src, m, acc=None):
    """亮态底：纸灰（向强调色偏 8%，Y≤150）+ 人物阈值灰阶 + 淡网格。"""
    acc = acc or RUST_ACC
    paper = np.array(PAPER_G, np.float32) * 0.92 + np.array(acc["main"], np.float32) * 0.08
    paper *= min(1.0, 148.0 / (0.299 * paper[0] + 0.587 * paper[1] + 0.114 * paper[2]))
    out = np.empty((H, W, 3), np.uint8); out[:] = paper.astype(np.uint8)
    s = int(48 * R)
    out[::s, :] = (paper * 0.92).astype(np.uint8); out[:, ::s] = (paper * 0.92).astype(np.uint8)
    pc = palette_c(src, m, acc)
    mb = (cv2.GaussianBlur(m, (0, 0), 1.5) > 0.5)[..., None]
    return np.where(mb, pc, out)


def lyric_serif(lay, jp, en, xy, size=58, col=(240, 236, 228, 255), sub=(200, 196, 188, 255)):
    x, y = xy
    if jp:
        im = mixed(jp, size * R, col, jp=F_JPS, en=F_WIDE, en_scale=0.62)
        lay.alpha_composite(im, (int(x), int(y))); y += im.height + 6 * R
    if en:
        im = mixed(en, 17 * R, sub, jp=F_JP, en=F_WIDE, track=1.5 * R)
        lay.alpha_composite(im, (int(x), int(y)))
    return lay


def compose(base, lay):
    import gpu
    if gpu.use():
        return gpu.compose(base, np.asarray(lay))
    b = Image.fromarray(base).convert("RGBA"); b.alpha_composite(lay)
    return np.asarray(b.convert("RGB"))


def luma(a):
    return float(cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)[..., 0].mean())


# ---------------------------------------------------------------- 风格帧
def load(name):
    return cv2.cvtColor(cv2.imread(os.path.join(QA, "src", name)), cv2.COLOR_BGR2RGB)


def split_lyric(text):
    """把一句歌词拆成 日文部分 / 英文部分（用于两行排版）。"""
    jp = "".join(s for c, s in runs(text) if c).strip()
    en = " ".join(s.strip() for c, s in runs(text) if not c and s.strip()).strip()
    return jp, en


def stills():
    from anime_det import Det
    FD, PD = Det("face"), Det("person")
    os.makedirs(QA, exist_ok=True)
    rep = []

    # R1：Saber Alter 面甲特写（fKoHcAM0PEk 17.2，HF 4K60）锁脸；歌词 8「翻す深紅」
    src = load("fKoHcAM0PEk_17.2.png")
    faces = FD(src, th=0.2); pers = PD(src)
    tgt = faces[0][:4] if faces else (pers[0][:4] if pers else (W * 0.35, H * 0.2, W * 0.6, H * 0.7))
    hud = HudR(tgt, faces, tags=["CLASS  SABER [ALTER]", "PARAM  STR A / MGI A++", "NP     EXCALIBUR MORGAN"],
               head="TARGET // SABER ALTER", seed=1, acc=accent_of(src))
    t = lyric(8)["start"] / 1000.0 + 0.4
    base = r_base(src, 30)
    lay = hud.overlay(30, t, beat_k=5, beat_f=3)
    lyric_hud(lay, lyric(8)["text"], 30, 0, (W * 0.60, H * 0.70), size=52, seed=8, acc=hud.acc)
    out = compose(base, lay)
    rep.append(("R1_saber_alter", out, faces, pers))

    # R2：Lancer 高举魔枪（5IiGdXV-wow 45.0）锁人物 + 波形撕裂；歌词 30
    src = load("5IiGdXV-wow_45.0.png")
    faces = FD(src, th=0.2); pers = PD(src)
    tgt = pers[0][:4] if pers else (W * 0.62, H * 0.3, W * 0.78, H * 0.95)
    hud = HudR(tgt, faces, tags=["CLASS  LANCER", "NP     GAE BOLG", "RANGE  2-4 / 1"], head="TARGET // LANCER", seed=2, acc=accent_of(src))
    t = lyric(30)["start"] / 1000.0 + 0.6
    base = wave_tear(r_base(src, 12), 0.45, tint=hud.acc["sec"])
    lay = hud.overlay(10, t, beat_k=9, beat_f=6)
    lyric_hud(lay, lyric(30)["text"], 40, 0, (W * 0.06, H * 0.62), size=44, seed=30, acc=hud.acc)
    out = compose(base, lay)
    rep.append(("R2_lancer_tear", out, faces, pers))

    # C1：贞德对镜（CQ8v15KMND4 45.2，Apocrypha OP1 4K60）暗态星座；歌词 45「逆説歪む黎明」
    src = load("CQ8v15KMND4_45.2.png")
    faces = FD(src, th=0.2); m = seg_mask(src)
    # 只要镜中贞德（右侧人物）：用脸所在连通域
    if faces:
        fx = (faces[0][0] + faces[0][2]) / 2
        lab_n, lab = cv2.connectedComponents((m > 0.5).astype(np.uint8))
        keep = lab[int((faces[0][1] + faces[0][3]) / 2), int(fx)]
        if keep:
            m = m * (lab == keep)
    nodes = nodes_from_mask(m, faces, n=7)
    labels = ["RULER", "LA PUCELLE", "BANNER", "ORLÉANS", "1412", "FAITH", "MIRROR", "SAINT"]
    base, lay = c_dark(src, m, 60, nodes, labels[:len(nodes)], acc=accent_of(src, m))
    jp, en = split_lyric(lyric(45)["text"])
    lyric_serif(lay, jp, "PARADOX // DISTORTED DAWN", (W * 0.05, H * 0.66), size=72)
    out = compose(base, lay)
    rep.append(("C1_jeanne_dark", out, faces, None))

    # C2：贞德 + 齐格侧脸（CQ8v15KMND4 88.6）亮态档案；歌词 26
    src = load("CQ8v15KMND4_88.6.png")
    faces = FD(src, th=0.2); m = seg_mask(src)
    fmask = face_mask(faces, 8)
    acc = accent_of(src, m)
    base = c_bright(src, m, acc)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    seeds = [(W * 0.30, H * 0.85, 360), (W * 0.62, H * 0.95, 260), (W * 0.40, H * 0.10, 200)]
    frac = pixel_growth(lay, m, fmask, 0.85, seeds, col=acc["main"])
    fs = sorted(faces, key=lambda b: b[0])
    titles = ["SIEG // HOMUNCULUS", "RULER // JEANNE D'ARC"]
    dsts = [((W * 0.04, H * 0.10), (int(560 * R), int(330 * R))), ((W * 0.04, H * 0.52), (int(560 * R), int(330 * R)))]
    for (x0, y0, x1, y1, _), title, (dst, size) in zip(fs, titles, dsts):
        w_, h_ = (x1 - x0), (y1 - y0) * 0.55                  # 只取眼睛一带（宽幅），比例跟框一致
        h_ = w_ * size[1] / size[0]
        crop_box(lay, src, (x0, y0 + (y1 - y0) * 0.22, w_, h_), dst, size, title, 1.0, acc=acc)
    d = ImageDraw.Draw(lay)
    tb = int(W * 0.30)
    d.rectangle([W - tb - 60 * R, 60 * R, W - 60 * R, 60 * R + 44 * R], fill=acc["main"] + (255,))
    d.text((W - tb - 48 * R, 66 * R), "ARCHIVE // APOCRYPHA", font=font(F_WIDE, 22 * R), fill=acc["txt"] + (255,))
    jp, en = split_lyric(lyric(26)["text"])
    lyric_serif(lay, jp, en.upper(), (W * 0.05, H * 0.86), size=60, col=INK + (255,), sub=(40, 38, 36, 255))
    out = compose(base, lay)
    rep.append(("C2_archive_bright", out, faces, None))
    print("accent area C2 = %.1f%%" % (frac * 100))

    for name, img, faces, pers in rep:
        cv2.imwrite(os.path.join(QA, name + ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(name, "meanY %.1f" % luma(img), "faces", len(faces or []), "persons", len(pers or []))
    sheet = np.vstack([np.hstack([cv2.resize(rep[0][1], (1280, 720)), cv2.resize(rep[1][1], (1280, 720))]),
                       np.hstack([cv2.resize(rep[2][1], (1280, 720)), cv2.resize(rep[3][1], (1280, 720))])])
    cv2.imwrite(os.path.join(QA, "_sheet_v2_stills.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def swatches():
    """每张源图的强调色色板（main / hi / dk / sec），检查取色是否跟画面走。"""
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(QA, "src", "*_*.png"))):
        src = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        a = accent_of(src)
        th = cv2.resize(src, (480, 270))
        sw = np.zeros((270, 240, 3), np.uint8)
        for i, k in enumerate(("main", "hi", "dk", "sec")):
            sw[:, i * 60:(i + 1) * 60] = a[k]
        rows.append(np.hstack([th, sw]))
        print(os.path.basename(p), "hue %.0f / %.0f" % (a["hue"], a["hue2"]), a["main"])
    while len(rows) % 3:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack([np.hstack(rows[i:i + 3]) for i in range(0, len(rows), 3)])
    cv2.imwrite(os.path.join(QA, "_accent_swatches.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    swatches() if "--swatches" in sys.argv else stills()

"""v8_post.py — Edgerunners-style post layer (generic, resolution-agnostic).

Ported and generalised from the Cyberpunk: Edgerunners AMV (v5 series) built on this engine.
Every function takes plain uint8/float32 RGB arrays plus optional face boxes (x0, y0, x1, y1 in pixels),
so it plugs into any `styled()` / `render_frame()` without touching v2_preview.
Nothing here is on by default: v2_preview behaviour is unchanged unless a caller imports this module.
No project paths, timings, lyrics or media are stored here — timings come from your EDL.
"""
import math
import numpy as np
import cv2

# ---------------------------------------------------------------- config / presets
CFG = dict(
    face_cap=225,          # hard QC ceiling: p95 luma of every face box after encode
    face_cap_inner=220,    # guard target before 4:2:0 encode (headroom for chroma bleed on saturated reds)
    face_grow=0.15,        # face box growth for masks
    speed_lines=False,     # radial speed lines are OFF by default (user rejected them repeatedly)
    grain=3.0,             # fine grain sigma, never on faces
    halftone_cell=10,
    max_flash_per_s=3,     # matches reference/08_qc.md
    min_flash_gap_s=0.34,
)

ACCENT = dict(cyan=(120.0, 212.0, 235.0), red=(212.0, 53.0, 75.0), ink=(14.0, 16.0, 22.0), paper=(238.0, 236.0, 240.0))

# duotone palettes: (shadow, mid, highlight)
PAL = dict(
    blue=((6, 12, 30), (40, 120, 214), (200, 236, 255)),
    red=((18, 6, 14), (214, 40, 58), (255, 200, 192)),
    purple=((14, 6, 32), (150, 62, 232), (236, 204, 255)),
    teal=((4, 16, 20), (30, 150, 160), (210, 250, 240)),
)

# lyric motion presets (code only — the words come from your own timing data, never stored here)
LYRIC_MOTION = {
    # v5a "hot word": key word punches in large, overshoots, settles, then wall-tiles faintly behind
    "hot": dict(in_s=0.10, scale0=1.6, scale1=1.0, overshoot=0.06, hold_alpha=1.0, out_s=0.18, wall=True, wall_alpha=0.12, shake_px=6),
    # v5f "insert style": serif italic key word + oversized ghost behind, slow drift, avoids faces
    "insert": dict(in_s=0.22, scale0=1.08, scale1=1.0, overshoot=0.0, hold_alpha=0.95, out_s=0.30, ghost=True, ghost_scale=2.2,
                   ghost_alpha=0.10, drift_px_s=14, avoid_faces=True),
    # plain line, per-word reveal following vocal onsets
    "line": dict(in_s=0.08, scale0=1.0, scale1=1.0, overshoot=0.0, hold_alpha=1.0, out_s=0.12),
}

# ---------------------------------------------------------------- helpers
def ss(u): u = float(np.clip(u, 0, 1)); return u * u * (3 - 2 * u)
def eo(u): u = float(np.clip(u, 0, 1)); return 1 - (1 - u) ** 3
def ei(u): u = float(np.clip(u, 0, 1)); return u ** 3
def expo_out(u): u = float(np.clip(u, 0, 1)); return 1.0 if u >= 1 else 1 - 2 ** (-10 * u)
def luma(a): return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
def _f(img): return img.astype(np.float32)
def _u8(f): return np.clip(f, 0, 255).astype(np.uint8)
def _grid(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); return yy, xx
def _ib(b, h, w):
    x0, y0, x1, y1 = b
    return max(0, int(x0)), max(0, int(y0)), min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))

# ---------------------------------------------------------------- face guard
def face_mask(shape, boxes, grow=None, sig=18):
    """soft mask (H, W) float32, 1 on (grown) faces"""
    h, w = shape[:2]; g = CFG["face_grow"] if grow is None else grow
    m = np.zeros((h, w), np.float32)
    for b in boxes or ():
        x0, y0, x1, y1 = _ib(b, h, w); dx, dy = int((x1 - x0) * g), int((y1 - y0) * g)
        m[max(0, y0 - dy):min(h, y1 + dy), max(0, x0 - dx):min(w, x1 + dx)] = 1
    return cv2.GaussianBlur(m, (0, 0), sig) if m.any() and sig > 0 else m

def face_guard(img, boxes, cap=None, feather=6):
    """final pass after lyrics/HUD: clamp every face box to `cap` (feathered) so encoded faces stay <= CFG['face_cap']"""
    if not boxes: return img
    cap = CFG["face_cap_inner"] if cap is None else cap
    h, w = img.shape[:2]; m = np.zeros((h, w), np.float32)
    for b in boxes:
        x0, y0, x1, y1 = _ib(b, h, w)
        if x1 > x0 and y1 > y0: m[y0:y1, x0:x1] = 1
    if feather:   # grow by 3*feather first so the whole box stays fully clamped after feathering
        k = 6 * feather + 1; m = np.minimum(1.0, cv2.GaussianBlur(cv2.dilate(m, np.ones((k, k), np.uint8)), (0, 0), feather) + m)
    m = m[..., None]
    o = _f(img); return _u8(o * (1 - m) + np.minimum(o, cap) * m)

# ---------------------------------------------------------------- grade / looks
def duo(f, p):
    L = np.clip(luma(f) / 255.0, 0, 1)[..., None]; d, m, hi = (np.array(x, np.float32) for x in p)
    lo = d + (m - d) * np.clip(L * 2, 0, 1); up = m + (hi - m) * np.clip(L * 2 - 1, 0, 1)
    return np.where(L < 0.5, lo, up)

def grade(img, kind, split_angle=0.42):
    """colour-block grade presets: cool / cool_l / red / purple / split / d2 ('' = passthrough). returns float32"""
    f = _f(img)
    if kind in ("cool", "cool_l"):
        g = luma(f)[..., None]; des = f * 0.45 + g * 0.55; k = 0.4 if kind == "cool" else 0.25
        return des * (1 - k) + duo(f, PAL["blue"]) * k
    if kind in ("red", "purple", "teal"): return f * 0.22 + duo(f, PAL[kind]) * 0.78
    if kind == "split":   # diagonal blue | red with a thin hot seam
        h, w = f.shape[:2]; yy, xx = _grid(h, w)
        e = (xx - w / 2) + split_angle * (yy - h / 2); m = np.clip(e / 3.0 + 0.5, 0, 1)[..., None]
        out = f * 0.3 + (duo(f, PAL["blue"]) * (1 - m) + duo(f, PAL["red"]) * m) * 0.7
        return out + np.exp(-(e ** 2) / 8.0)[..., None] * 120
    if kind == "d2":      # night-city look: lifted teal shadows, warm-neutral highs, gentle S-curve, keeps line detail
        L = luma(f)[..., None] / 255.0
        f = f + (1 - L) ** 2 * np.array((-6, 8, 14), np.float32) + L ** 2 * np.array((10, 2, -8), np.float32)
        x = np.clip(f / 255.0, 0, 1); x = x + 0.18 * (x - 0.5) * (1 - np.abs(2 * x - 1))
        return x * 255.0
    return f

def color_jump(img, kinds, t, period):
    """grade-jump: cycle colour-block grades every `period` seconds (one jump = one flash for QC)"""
    return grade(img, kinds[int(t // period) % len(kinds)])

# ---------------------------------------------------------------- light FX
def light_glow(f, src, fmask, k=0.8, burst=0.0, big=False, col=ACCENT["cyan"]):
    """local anamorphic streak + point glow on small highlights of the pre-grade source; faces: specular dots only"""
    h, w = f.shape[:2]; q = cv2.resize(src, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32); L = luma(q)
    fq = cv2.resize(fmask, (w // 4, h // 4)) > 0.3
    thr = max(160.0, float(np.percentile(L, 99.6))); thf = max(196.0, float(np.percentile(L[fq], 99.3))) if fq.any() else 999.0
    m = ((L > thr) & ~fq) | ((L > thf) & fq)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = np.zeros(n, bool); amax = 6000 if big else 700
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]; keep[i] = a <= (40 if fq[lab == i].mean() > 0.5 else amax)
    m = keep[lab].astype(np.float32)
    if m.sum() < 1: return f
    s = cv2.blur(m, (161, 1)) * 5.0 + cv2.blur(m, (41, 1)) * 2.0 + cv2.GaussianBlur(m, (0, 0), 5) * 2.2 + cv2.GaussianBlur(m, (0, 0), 1.5) * 0.8
    if burst > 0:
        for kk in (np.eye(61, dtype=np.float32), np.fliplr(np.eye(61, dtype=np.float32)), np.ones((61, 1), np.float32)):
            s += cv2.filter2D(m, -1, kk / 61.0) * 6.0 * burst
    s = np.clip(cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1.6)[..., None]
    return f + s * np.array(col, np.float32) * k

_DOT = {}
def halftone(f, fmask, alpha, cell=None, ink=(12, 10, 20)):
    """luma-driven dot screen; never on faces, skips near-black and near-white"""
    cell = cell or CFG["halftone_cell"]; h, w = f.shape[:2]; key = (h, w, cell)
    if key not in _DOT:
        yy, xx = _grid(h, w); cy, cx = (yy % cell) - cell / 2 + 0.5, (xx % cell) - cell / 2 + 0.5; _DOT[key] = np.sqrt(cx * cx + cy * cy)
    L = cv2.resize(cv2.resize(luma(f), (w // cell, h // cell), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_NEAREST) / 255.0
    d = np.clip((1 - L) * cell * 0.62 - _DOT[key] + 0.5, 0, 1)
    a = (alpha * d * (1 - fmask) * (L > 0.12) * (L < 0.85))[..., None]
    return f * (1 - a) + np.array(ink, np.float32) * a

def rings(f, fmask, u, c, alpha, col=(250, 236, 240), life=0.9, speed=1500):
    """shock rings; u = seconds since the hit"""
    if u < 0 or u > life: return f
    h, w = f.shape[:2]; yy, xx = _grid(h, w); r = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2); acc = np.zeros((h, w), np.float32)
    for k in range(6):
        R = u * speed * w / 2560 + k * 110 * w / 2560
        acc = np.maximum(acc, np.exp(-((r - R) ** 2) / (2 * (2.5 + k * 0.6) ** 2)) * (1 - k / 7.0))
    a = (alpha * (1 - u / life) * acc * (1 - fmask))[..., None]
    return f * (1 - a) + np.array(col, np.float32) * a

def speed_lines(f, fmask, F, c, r0, alpha, col=(240, 236, 244), force=False):
    """radial speed lines — no-op unless CFG['speed_lines'] or force=True"""
    if not (CFG["speed_lines"] or force): return f
    h, w = f.shape[:2]; yy, xx = _grid(h, w); ang = np.arctan2(yy - c[1], xx - c[0]); r = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
    rng = np.random.RandomState(1000 + F // 2); N = 180; on = rng.rand(N) < 0.33; wd = rng.rand(N) * 0.35 + 0.08
    u = (ang + np.pi) / (2 * np.pi) * N; b = np.floor(u).astype(np.int32) % N; fr = u - np.floor(u)
    ln = on[b] & (np.abs(fr - 0.5) < wd[b] * 0.5); rr = np.clip((r - r0 * (0.8 + 0.4 * rng.rand(N)[b])) / 160.0, 0, 1)
    a = (alpha * ln * rr * (1 - fmask))[..., None]
    return f * (1 - a) + np.array(col, np.float32) * a

def finish(f, fmask, F, grain=None):
    """light halation + fine grain (grain never on faces)"""
    h, w = f.shape[:2]; L = luma(f); hb = cv2.GaussianBlur(np.clip(L - 190, 0, 65) / 65.0, (0, 0), 22)[..., None]
    f = f + hb * np.array((30, 10, 9), np.float32)
    s = CFG["grain"] if grain is None else grain
    if s > 0:
        g = np.random.RandomState(F).normal(0, s, (h // 2, w // 2)).astype(np.float32)
        f = f + cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)[..., None] * (1 - fmask[..., None])
    return f

def flash(img, u, mode="invert", life=0.12, shake_px=14, seed=0):
    """invert / BW flash + shake; u = seconds since the hit. counts as one flash for QC"""
    if u < 0 or u > life + 0.2: return img
    f = _f(img); h, w = f.shape[:2]
    if u <= life:
        k = 1 - u / life
        alt = 255 - f if mode == "invert" else np.repeat(np.clip((luma(f)[..., None] - 70) * 2.2, 0, 255), 3, -1)
        f = f * (1 - k) + alt * k
    a = shake_px * (1 - eo(u / (life + 0.2))); r = np.random.RandomState(seed + int(u * 240))
    M = np.float32([[1, 0, r.uniform(-a, a)], [0, 1, r.uniform(-a, a)]])
    return _u8(cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT))

def push(img, z):
    """slow push-in (z >= 1)"""
    if z <= 1.0005: return img
    h, w = img.shape[:2]; M = np.float32([[z, 0, (1 - z) * w / 2], [0, z, (1 - z) * h / 2]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ---------------------------------------------------------------- impact frame + slam
def impact_frame(img, fmask=None, thr=118, ink=ACCENT["ink"], paper=(246, 242, 236), accent=None):
    """2-tone ink impact frame (threshold + line boost); faces keep a softer threshold so they never go flat white"""
    f = _f(img); L = luma(f); edges = cv2.Canny(img if img.dtype == np.uint8 else _u8(f), 60, 160) > 0
    t = np.full(L.shape, float(thr), np.float32)
    if fmask is not None: t = t * (1 - fmask) + (thr * 0.75) * fmask
    m = ((L > t) & ~edges).astype(np.float32)[..., None]
    out = np.array(ink, np.float32) * (1 - m) + np.array(paper if accent is None else accent, np.float32) * m
    return _u8(out)

def slam(img, u, z0=1.35, dur=0.12, shake_px=10, seed=0):
    """scale slam into the frame (z0 -> 1 over dur) with decaying shake; u = seconds since hit"""
    h, w = img.shape[:2]; z = 1 + (z0 - 1) * (1 - eo(u / dur)) if u < dur else 1.0
    a = shake_px * max(0.0, 1 - u / (dur * 3)); r = np.random.RandomState(seed + int(u * 240))
    M = np.float32([[z, 0, (1 - z) * w / 2 + r.uniform(-a, a)], [0, z, (1 - z) * h / 2 + r.uniform(-a, a)]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ---------------------------------------------------------------- panel stack / grid zoom-out / shards
def paste_panel(dst, src, cx, cy, pw, rot, border=8, shadow=True, paper=ACCENT["paper"]):
    """paste src as a bordered tilted panel (16:9 of width pw) centred at (cx, cy); never upscales beyond source"""
    h, w = dst.shape[:2]; ph = int(pw * src.shape[0] / src.shape[1])
    p = cv2.resize(src, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)
    p = cv2.copyMakeBorder(p, border, border, border, border, cv2.BORDER_CONSTANT, value=paper)
    H2, W2 = p.shape[:2]; M = cv2.getRotationMatrix2D((W2 / 2, H2 / 2), rot, 1.0); M[0, 2] += cx - W2 / 2; M[1, 2] += cy - H2 / 2
    a = cv2.warpAffine(np.ones((H2, W2), np.float32), M, (w, h)); im = cv2.warpAffine(p, M, (w, h)); d = _f(dst)
    if shadow:
        sh = cv2.GaussianBlur(np.roll(np.roll(a, 14, 0), 14, 1), (0, 0), 10)[..., None] * 0.55; d = d * (1 - sh) + np.array((8, 8, 14), np.float32) * sh
    a = a[..., None]; return d * (1 - a) + im * a

def panel_stack(base, stills, layout, t, starts, pop=0.25, pop_s=0.1):
    """panels drop in one per beat: layout = [(nx, ny, width_frac, rot), ...]; starts = beat times"""
    f = _f(base); h, w = f.shape[:2]
    for im, (nx, ny, wf, rot), t0 in zip(stills, layout, starts):
        if t >= t0: z = 1 + pop * (1 - eo((t - t0) / pop_s)); f = paste_panel(f, im, nx * w, ny * h, int(w * wf * z), rot)
    return f

def grid_zoom(cur, tiles, z, n=3, gutter=8, bg=(10, 10, 16)):
    """n x n grid with `cur` in the centre; z = n -> full frame, z = 1 -> whole grid visible (zoom-out reveal)"""
    h, w = cur.shape[:2]
    if z >= n - 1e-3: return _f(cur)
    out = np.full((h, w, 3), bg, np.float32); g = gutter * z / n; mid = (n * n) // 2; ti = iter(tiles)
    for i in range(n * n):
        src = cur if i == mid else next(ti); r, c = divmod(i, n)
        tw, th = w / n * z, h / n * z; x0 = w / 2 + (c - n / 2) * tw; y0 = h / 2 + (r - n / 2) * th
        M = np.float32([[(tw - 2 * g) / w, 0, x0 + g], [0, (th - 2 * g) / h, y0 + g]])
        im = cv2.warpAffine(_f(src), M, (w, h), flags=cv2.INTER_AREA); a = cv2.warpAffine(np.ones((h, w), np.float32), M, (w, h))[..., None]
        out = out * (1 - a) + im * a
    return out

def grid_zoom_curve(u, n=3, hold=0.03):
    """in (n->1) / settle / out (1->n) curve over u in [0, 1]"""
    if u < 1 / 3: return n - (n - 1) * eo(u * 3)
    if u < 2 / 3: return 1.0 - hold * ss((u - 1 / 3) * 3)
    return (1 - hold) + (n - 1 + hold) * ei((u - 2 / 3) * 3)

def diagonal_shards(A, B, u, angle_deg=62, dist=None, seam=200):
    """A splits along a diagonal and the halves fly apart revealing B; u in [0, 1]"""
    h, w = A.shape[:2]; yy, xx = _grid(h, w); th = math.radians(angle_deg)
    n = np.array((math.cos(th), math.sin(th)), np.float32); tg = np.array((-n[1], n[0]), np.float32)
    side = ((xx - w / 2) * n[0] + (yy - h / 2) * n[1]) > 0; d = (max(float(u), 0.0) ** 1.7) * (dist or w * 0.43); out = _f(B)
    for sgn, msk in ((1, side), (-1, ~side)):
        o = (n * 0.75 + tg * 0.35) * d * sgn; M = np.float32([[1, 0, o[0]], [0, 1, o[1]]])
        a = cv2.warpAffine(msk.astype(np.float32), M, (w, h))[..., None]; im = cv2.warpAffine(_f(A), M, (w, h)); out = out * (1 - a) + im * a
    if u < 0.5:
        e = np.abs((xx - w / 2) * n[0] + (yy - h / 2) * n[1]); out += (np.exp(-(e ** 2) / 18.0) * (1 - u * 2) * seam)[..., None]
    return out

# ---------------------------------------------------------------- overlap match-cut transitions (v5g)
def _band(d, hw): return np.clip(hw + 0.5 - np.abs(d), 0, 1)

def _face_safe_mix(A, B, al, fa=(), fb=(), deco=None, col=ACCENT["cyan"]):
    """mix by coverage `al`; inside any face box coverage = box mean (no wipe edge across a face); deco off faces"""
    h, w = A.shape[:2]; al = al.astype(np.float32).copy(); fm = np.zeros((h, w), np.float32)
    for b in list(fa) + list(fb):
        x0, y0, x1, y1 = _ib(b, h, w)
        if x1 > x0 and y1 > y0: al[y0:y1, x0:x1] = al[y0:y1, x0:x1].mean(); fm[y0:y1, x0:x1] = 1
    out = _f(A) * (1 - al[..., None]) + _f(B) * al[..., None]
    if deco is not None: d = (deco * (1 - fm))[..., None]; out = out * (1 - d) + np.array(col, np.float32) * d
    return out

def ov_split(A, B, u, angle=0.16, line=2.2, fa=(), fb=()):
    """slanted split wipe with a thin accent seam; u in [0, 1]"""
    h, w = A.shape[:2]; yy, xx = _grid(h, w); x = -w * 0.2 + expo_out(u) * w * 1.4
    d = xx - (x + angle * (yy - h / 2)); al = np.clip(-d + 0.5, 0, 1)
    return _face_safe_mix(A, B, al, fa, fb, _band(d, line) * (1 - u))

def ov_match(A, B, u, box_a, box_b, fa=(), fb=()):
    """match-cut overlap: B is scaled/translated so its subject box lands on A's, then eases to identity while A fades"""
    h, w = A.shape[:2]
    ca = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2); cb = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
    s0 = (box_a[2] - box_a[0]) / max(1.0, box_b[2] - box_b[0]); k = 1 - ss(u); s = 1 + (s0 - 1) * k
    M = np.float32([[s, 0, (ca[0] - s * cb[0]) * k], [0, s, (ca[1] - s * cb[1]) * k]])
    Bt = cv2.warpAffine(B, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return _face_safe_mix(A, Bt, np.full((h, w), ss(u), np.float32), fa, fb)

def ov_frames(A, B, u, n=3, slant=0.16, gutter=14, line=2.2, stagger=0.12, col=ACCENT["ink"], fa=(), fb=()):
    """A splits into n slanted frames that slide off one by one (right first) revealing B"""
    h, w = A.shape[:2]; yy, xx = _grid(h, w); xs = [-w] + [w * (i + 1) / n for i in range(n - 1)] + [2 * w]
    At = np.zeros((h, w, 3), np.float32); m = np.zeros((h, w), np.float32); deco = np.zeros((h, w), np.float32)
    for k in range(n):
        uk = expo_out((u - (n - 1 - k) * stagger) / max(1e-3, 1 - (n - 1) * stagger)); dx = -uk * w * 1.15 * (1 + 0.15 * k)
        Ak = cv2.warpAffine(A, np.float32([[1, 0, dx], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        yc = yy - h / 2; dl = xx - (xs[k] + dx + gutter + slant * yc); dr = (xs[k + 1] + dx - gutter + slant * yc) - xx
        mk = np.clip(dl + 0.5, 0, 1) * np.clip(dr + 0.5, 0, 1)
        At = At * (1 - mk[..., None]) + Ak * mk[..., None]; m = np.maximum(m, mk); deco = np.maximum(deco, _band(np.minimum(dl, dr), line) * (1 - uk))
    return _face_safe_mix(B, At, m, fb, fa, deco, col)

# ---------------------------------------------------------------- stacked PiP colour-jump (v5h)
def _hit(r1, r2): return not (r1[0] + r1[2] <= r2[0] or r2[0] + r2[2] <= r1[0] or r1[1] + r1[3] <= r2[1] or r2[1] + r2[3] <= r1[1])

def pip_place(want, zones, side=None, step=0.01):
    """find the face-free normalised box (x, y, w, h) closest to `want`, shrinking if needed; zones = normalised face/hair boxes"""
    x, y, w0, h0 = want; best = None
    for kw in (1, .9, .8, .7, .6, .5):
        for kh in (1, .9, .8, .7):
            w, h = w0 * kw, h0 * kh
            for nx in np.arange(0.01, 0.99 - w + 1e-6, step):
                for ny in np.arange(0.02, 0.98 - h + 1e-6, step * 2):
                    c = (float(nx), float(ny), w, h)
                    if any(_hit(c, z) for z in zones): continue
                    sc = abs(nx - x) + abs(ny - y) + 0.5 * (abs(w - w0) + abs(h - h0))
                    if side is not None: sc += 0.35 * (((nx + w / 2) > 0.5) != bool(side))
                    if best is None or sc < best[0]: best = (sc, c)
        if best: return best[1]
    return None

def pip_stack(A, layers, warm=None, warm_gain=(1.10, 0.96, 0.88), fa=(), cap=None):
    """A + stacked picture-in-picture windows; layers = [(box_norm, img_full_frame), ...] drawn in order.
    warm = normalised box tinted for the colour-jump; faces in A are clamped to `cap` so the tint never blows them out"""
    h, w = A.shape[:2]; fr = _f(A); cap = CFG["face_cap"] if cap is None else cap
    if warm:
        x0, y0 = int(warm[0] * w), int(warm[1] * h); x1, y1 = x0 + int(warm[2] * w), y0 + int(warm[3] * h)
        fr[y0:y1, x0:x1] *= np.array(warm_gain, np.float32)
        for b in fa: X0, Y0, X1, Y1 = _ib(b, h, w); fr[Y0:Y1, X0:X1] = np.minimum(fr[Y0:Y1, X0:X1], cap)
    for (bx, by, bw, bh), src in layers:
        x0, y0 = int(bx * w), int(by * h); x1, y1 = min(w, x0 + int(bw * w)), min(h, y0 + int(bh * h)); fr[y0:y1, x0:x1] = _f(src)[y0:y1, x0:x1]
    return _u8(fr)

def pip_cut_gain(j, hold, post, g0=0.88):
    """after the hard cut to B: hold gain g0 for `hold` frames, then ease to 1 by `post`"""
    return g0 if j < hold else g0 + (1 - g0) * ss((j - hold + 1) / max(1, post - hold))

# ---------------------------------------------------------------- manga page / panel transition
def free_panels(w, h, n=5, seed=0, gutter=18, slant=0.08):
    """sharp free-panel page layout: list of 4-point polygons (pixels). deterministic per seed"""
    r = np.random.RandomState(seed); rows = 2 if n <= 4 else 3; polys = []; ys = np.linspace(0, h, rows + 1)
    per = [n // rows + (1 if i < n % rows else 0) for i in range(rows)]
    for ri in range(rows):
        k = per[ri]; xs = np.linspace(0, w, k + 1); xs[1:-1] += r.uniform(-0.08, 0.08, k - 1) * w
        sl = r.uniform(-slant, slant) * h
        for ci in range(k):
            x0, x1 = xs[ci] + gutter / 2, xs[ci + 1] - gutter / 2; y0, y1 = ys[ri] + gutter / 2, ys[ri + 1] - gutter / 2
            polys.append(np.float32([[x0 + sl * (ri > 0), y0], [x1 + sl * (ri > 0), y0], [x1 - sl, y1], [x0 - sl, y1]]))
    return polys

def manga_page(frames, polys, paper=ACCENT["paper"], line=3, ink=ACCENT["ink"], tone=None):
    """compose frames (full-res RGB) into sharp panels: INTER_AREA downscale only, crisp 1-px-AA borders, optional duo tone"""
    h, w = frames[0].shape[:2]; page = np.full((h, w, 3), paper, np.float32)
    for im, p in zip(frames, polys):
        x0, y0 = p[:, 0].min(), p[:, 1].min(); x1, y1 = p[:, 0].max(), p[:, 1].max(); pw, ph = max(2, int(x1 - x0)), max(2, int(y1 - y0))
        s = max(pw / im.shape[1], ph / im.shape[0]); src = cv2.resize(im, (int(im.shape[1] * s) + 1, int(im.shape[0] * s) + 1),
                                                                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        ox, oy = (src.shape[1] - pw) // 2, (src.shape[0] - ph) // 2; tile = _f(src[oy:oy + ph, ox:ox + pw])
        if tone: tile = duo(tile, tone)
        m = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(m, p.astype(np.int32), 255, cv2.LINE_AA)
        X0, Y0 = int(x0), int(y0); cx0, cy0 = max(0, X0), max(0, Y0); cx1, cy1 = min(w, X0 + tile.shape[1]), min(h, Y0 + tile.shape[0])
        canvas = np.zeros((h, w, 3), np.float32); canvas[cy0:cy1, cx0:cx1] = tile[cy0 - Y0:cy1 - Y0, cx0 - X0:cx1 - X0]
        a = (m.astype(np.float32) / 255)[..., None]; page = page * (1 - a) + canvas * a
        cv2.polylines(page, [p.astype(np.int32)], True, ink, line, cv2.LINE_AA)
    return page

def manga_transition(A, page, u, panel_poly, mode="push"):
    """dynamic panel transition: full-frame A shrinks into its panel on the page (u 0->1). mode push|pull|panL"""
    h, w = A.shape[:2]; p = panel_poly; x0, y0 = p[:, 0].min(), p[:, 1].min(); x1, y1 = p[:, 0].max(), p[:, 1].max(); k = eo(u)
    sx = 1 + ((x1 - x0) / w - 1) * k; sy = 1 + ((y1 - y0) / h - 1) * k
    tx, ty = x0 * k, y0 * k
    if mode == "pull": tx += (1 - k) * w * 0.04
    if mode == "panL": tx -= (1 - k) * w * 0.08
    M = np.float32([[sx, 0, tx], [0, sy, ty]]); At = cv2.warpAffine(_f(A), M, (w, h), flags=cv2.INTER_AREA)
    a = cv2.warpAffine(np.ones((h, w), np.float32), M, (w, h))[..., None]; return _f(page) * (1 - a) + At * a

# ---------------------------------------------------------------- fine HUD
def hud_fine(img, t, color=ACCENT["cyan"], alpha=0.85, margin=0.035, ticks=48, seed=0, labels=()):
    """thin precision HUD: corner brackets, tick ruler, moving scan marker, small labels (only verified text!)"""
    f = _u8(_f(img)).copy(); h, w = f.shape[:2]; m = int(min(w, h) * margin); L = int(min(w, h) * 0.06); c = tuple(int(v) for v in color)
    ov = f.copy()
    for (x, y, dx, dy) in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
        cv2.line(ov, (x, y), (x + dx * L, y), c, 1, cv2.LINE_AA); cv2.line(ov, (x, y), (x, y + dy * L), c, 1, cv2.LINE_AA)
    y = h - m - 6
    for i in range(ticks + 1):
        x = int(m + L + (w - 2 * m - 2 * L) * i / ticks); cv2.line(ov, (x, y), (x, y - (8 if i % 4 == 0 else 4)), c, 1, cv2.LINE_AA)
    sx = int(m + L + (w - 2 * m - 2 * L) * ((t * 0.37 + seed * 0.13) % 1.0)); cv2.line(ov, (sx, y - 14), (sx, y + 2), c, 1, cv2.LINE_AA)
    for i, s in enumerate(labels):
        cv2.putText(ov, s, (m + 6, m + L + 18 + 16 * i), cv2.FONT_HERSHEY_PLAIN, 0.9, c, 1, cv2.LINE_AA)
    return cv2.addWeighted(ov, alpha, f, 1 - alpha, 0)

# ---------------------------------------------------------------- letterbox joins / J-cut
def letterbox(img, ratio=2.39, u=1.0, color=(0, 0, 0)):
    """animated scope bars (u 0 -> 1 closes to `ratio`)"""
    f = img.copy(); h, w = f.shape[:2]; bar = int(round(max(0.0, (h - w / ratio) / 2) * ss(u)))
    if bar > 0: f[:bar] = color; f[h - bar:] = color
    return f

def letterbox_join(A, B, u, ratio=2.39):
    """join across a cut: bars close on A (u 0..0.5), cut at 0.5, open on B (0.5..1) — hides the edit, never a black frame"""
    return letterbox(A, ratio, u * 2) if u < 0.5 else letterbox(B, ratio, (1 - u) * 2)

def jcut(cut_frame, lead_s, fps):
    """J-cut: audio of B starts `lead_s` before the picture cut. returns (audio_in_frame, video_cut_frame)"""
    return int(round(cut_frame - lead_s * fps)), int(cut_frame)

# ---------------------------------------------------------------- lyric motion
def lyric_motion(u_in, u_out, preset="hot"):
    """per-word transform for a lyric preset: returns dict(scale, alpha); u_in = s since word on, u_out = s until word off"""
    p = LYRIC_MOTION[preset]; a = ss(u_in / p["in_s"]) * p["hold_alpha"]
    if u_out < p["out_s"]: a *= ss(u_out / p["out_s"])
    k = eo(u_in / p["in_s"]); s = p["scale0"] + (p["scale1"] - p["scale0"]) * k
    if p.get("overshoot") and u_in < p["in_s"] * 2: s -= p["overshoot"] * math.sin(math.pi * min(1, u_in / (p["in_s"] * 2)))
    return dict(scale=float(s), alpha=float(a))

# ---------------------------------------------------------------- QC
def qc_face(img, boxes, cap=None):
    """p95 luma of each face box; returns (ok, [p95...])"""
    cap = CFG["face_cap"] if cap is None else cap; h, w = img.shape[:2]; vals = []
    for b in boxes or ():
        x0, y0, x1, y1 = _ib(b, h, w)
        if x1 > x0 and y1 > y0: vals.append(float(np.percentile(luma(_f(img[y0:y1, x0:x1])), 95)))
    return all(v <= cap for v in vals), vals

def qc_flashes(luma_series, fps, jump=40.0):
    """flash events = |dLuma| >= jump; reports windows breaking max_flash_per_s or min_flash_gap_s"""
    L = np.asarray(luma_series, np.float32); ev = np.where(np.abs(np.diff(L)) >= jump)[0] + 1; bad = []
    for i, e in enumerate(ev):
        n = int(np.sum((ev >= e) & (ev < e + fps)))
        if n > CFG["max_flash_per_s"]: bad.append(("rate", int(e), n))
        if i and (e - ev[i - 1]) / fps < CFG["min_flash_gap_s"] and e - ev[i - 1] > 1: bad.append(("gap", int(e), float((e - ev[i - 1]) / fps)))
    return dict(events=ev.tolist(), violations=bad, black=[int(i) for i in np.where(L < 4)[0]])

# ---------------------------------------------------------------- smoke test
if __name__ == "__main__":
    W, H = 640, 360; r = np.random.RandomState(1)
    A = _u8(cv2.GaussianBlur(r.rand(H, W, 3).astype(np.float32) * 255, (0, 0), 6) * 1.4); B = 255 - A
    fb = [(260, 90, 380, 230)]; fm = face_mask(A.shape, fb)
    for k in ("", "cool", "cool_l", "red", "purple", "split", "d2"): assert grade(A, k).shape == A.shape
    f = light_glow(grade(A, "cool"), A, fm, burst=1.0); f = halftone(f, fm, 0.5); f = rings(f, fm, 0.2, (W / 2, H / 2), 0.7)
    assert np.array_equal(speed_lines(f, fm, 0, (W / 2, H / 2), 100, 0.5), f), "speed lines must default OFF"
    f = finish(f, fm, 0); g = face_guard(_u8(f + 80), fb); ok, v = qc_face(g, fb); assert ok, v
    for u in (0, 0.3, 0.7, 1): ov_split(A, B, u, fa=fb); ov_frames(A, B, u, fa=fb); ov_match(A, B, u, fb[0], (100, 50, 200, 150))
    diagonal_shards(A, B, 0.4); grid_zoom(A, [B] * 8, grid_zoom_curve(0.5)); panel_stack(A, [B, A], [(.2, .25, .3, -3), (.8, .7, .3, 4)], 1.0, [0, .5])
    impact_frame(A, fm); slam(A, 0.05); flash(A, 0.05); flash(A, 0.05, "bw")
    z = pip_place((0.6, 0.1, 0.3, 0.5), [(260 / W, 90 / H, 120 / W, 140 / H)]); assert z is not None
    pip_stack(A, [(z, B)], warm=(0, 0, .5, 1), fa=fb)
    page = manga_page([A, B, A, B, A], free_panels(W, H, 5)); manga_transition(A, page, 0.5, free_panels(W, H, 5)[2])
    hud_fine(A, 1.0, labels=("SAMPLE",)); letterbox_join(A, B, 0.3); assert jcut(600, 0.5, 60) == (570, 600)
    lyric_motion(0.05, 1.0, "hot"); lyric_motion(0.3, 0.1, "insert")
    q = qc_flashes([100] * 30 + [200, 100] * 4 + [100] * 30, 60); assert q["violations"]
    print("v8_post smoke OK")

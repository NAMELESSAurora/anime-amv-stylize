#!/usr/bin/env python3
"""Standalone reproduction of the "old G11" manga B&W look (numpy + opencv only).

  python3 manga_bw_g11.py IN.png OUT.png [--panel x,y,w,h | --panel-mask m.png] [--accent mask.png] [--protect m.png] [--pitch PX] [--legacy]
Default = team-agreed production settings (pitch 20 px @4K / 10 px @1080p, min dot 3 px @4K, clear centre fade, panel grain).
--legacy = the old G11 reference look described below (pitch 6 px @1080p).

Legacy pipeline (values from the original glitch_pass.py / filters.py):
  1. g = BGR->gray (cv2.COLOR_BGR2GRAY)
  2. margin (outside the panel): AM halftone of g, dot pitch 6 px @1080p, 45 deg grid in SCREEN coords,
     dot radius ~ darkness (1 - g/255)^1.0, binary 0/255 (pure black ink on pure white)
  3. panel interior: desaturate + contrast 1.1 around 128 (no dots, no thresholding)
  4. accent (optional mask, e.g. pupil reflections): rust LUT (B,G,R) = (0.35g, 0.55g+10, 0.95g+25),
     blobs darkened x0.65 where low-freq noise > 0.4, then 35% mixed back to gray ("mute"), blended by the soft mask
The radial edge->centre fade of the margin dots is NOT computed here: it comes from the darker vignette of the input paper.
Use --vignette S to synthesise one if your input has a flat paper (inferred option, not part of the original)."""
import argparse, cv2, numpy as np

def halftone(g, cell=6.0, angle=45.0, gamma=1.0):
    h, w = g.shape; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); a = np.deg2rad(angle)
    u = xx * np.cos(a) + yy * np.sin(a); v = -xx * np.sin(a) + yy * np.cos(a)
    du = (u % cell) - cell / 2; dv = (v % cell) - cell / 2
    d = np.sqrt(du ** 2 + dv ** 2) / (cell * 0.7071)          # 0 at dot centre .. 1 at cell corner
    dark = (1 - g.astype(np.float32) / 255) ** gamma
    return np.where(d < dark, 0, 255).astype(np.uint8)

def rust_accent(img, seed=21):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); h, w = g.shape
    lut = np.stack([g * 0.35, g * 0.55 + 10, g * 0.95 + 25], -1)
    blob = cv2.resize(np.random.default_rng(seed).normal(0, 1, (h // 40 + 2, w // 40 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    lut = np.clip(lut * (1 - 0.35 * (blob > 0.4)[..., None]), 0, 255).astype(np.uint8)
    L = cv2.cvtColor(cv2.cvtColor(lut, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(lut, 0.65, L, 0.35, 0)

# ---- production defaults (team-agreed 4K values; everything scales with output height / 2160) ----
PROD_PITCH_4K = 20.0        # margin dot pitch px @4K (10 px @1080p)
PROD_MIN_DOT_4K = 3.0       # drop dots with diameter < 3 px @4K (1.5 px @1080p): no grey haze
PROD_GRAIN_4K = 2.5         # panel grain size px @4K (2-3 px)
PROD_GRAIN_AMP = 6.0        # grain std in luma levels (proposed default, tune by eye)
PROD_CLEAR_R = 0.45         # margin fade: 100% clear inside this normalised radius (proposed default)
PROD_FULL_R = 1.15          # ... full input-driven tone at/after this radius (corners), smoothstep in between (proposed default)
# ---- legacy = old G11 reference look (values from the original code) ----
LEGACY_PITCH_1080 = 6.0

def radial_fade(h, w, r0, r1):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    t = np.clip((r - r0) / max(r1 - r0, 1e-3), 0, 1); return t * t * (3 - 2 * t)

def halftone_prod(g, pitch, min_dot, fade):
    """45 deg AM dots in screen/paper coords; dot area = darkness * fade; dots below min_dot diameter removed"""
    h, w = g.shape; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); a = np.deg2rad(45)
    u = xx * np.cos(a) + yy * np.sin(a); v = -xx * np.sin(a) + yy * np.cos(a)
    du = (u % pitch) - pitch / 2; dv = (v % pitch) - pitch / 2; dist = np.sqrt(du ** 2 + dv ** 2)
    cov = np.clip((1 - g.astype(np.float32) / 255) * fade, 0, 1)
    rad = pitch * np.sqrt(cov / np.pi)                 # round dot with area = cov * pitch^2 (merges into a net above ~78%)
    rad = np.where(2 * rad < min_dot, 0, rad)
    return np.where(dist < rad, 0, 255).astype(np.uint8)

def manga_bw(img, panel=None, accent=None, pitch=None, ref_h=1080, vignette=0.0, legacy=False, seed=0):
    h, w = img.shape[:2]; s = h / 2160.0
    if legacy: pitch = pitch or LEGACY_PITCH_1080 * h / ref_h
    else: pitch = pitch or PROD_PITCH_4K * s
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if vignette > 0:   # optional (inferred): darken the paper towards the edges so the dots fade in from the border
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        g = np.clip(g * (1 - vignette * r ** 2), 0, 255).astype(np.uint8)
    if legacy: margin = halftone(g, pitch)
    else: margin = halftone_prod(g, pitch, PROD_MIN_DOT_4K * s, radial_fade(h, w, PROD_CLEAR_R, PROD_FULL_R))
    margin = cv2.cvtColor(margin, cv2.COLOR_GRAY2BGR)
    f = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) - 128) * 1.1 + 128
    if not legacy:   # light grain, 2-3 px @4K, fixed seed (static per frame unless you change it)
        n = np.random.default_rng(seed).normal(0, 1, (h, w)).astype(np.float32); sg = PROD_GRAIN_4K * s / 2
        n = cv2.GaussianBlur(n, (0, 0), sg); n *= PROD_GRAIN_AMP / max(n.std(), 1e-6); f = f + n
    face = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pm = np.zeros((h, w), np.uint8)
    if panel is None: pm[:] = 255
    elif isinstance(panel, np.ndarray): pm = panel
    else: x, y, pw, ph = panel; pm[y:y + ph, x:x + pw] = 255
    pm = cv2.dilate(pm, np.ones((3, 3), np.uint8))
    out = np.where((pm > 128)[..., None], face, margin)
    if accent is not None:
        k = accent.astype(np.float32)[..., None] / 255.0
        out = (out * (1 - k) + rust_accent(img) * k).astype(np.uint8)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('output')
    ap.add_argument('--panel', help='x,y,w,h of the image panel (default: whole frame = panel, no margin dots)')
    ap.add_argument('--panel-mask', help='grayscale mask image of the panel (alternative to --panel)')
    ap.add_argument('--margin-only', action='store_true', help='treat the whole frame as margin (dots everywhere)')
    ap.add_argument('--accent', help='grayscale soft mask for the rust accent (e.g. pupil reflection)')
    ap.add_argument('--protect', help='grayscale soft mask restored from the ORIGINAL input (e.g. lyric card / text)')
    ap.add_argument('--pitch', type=float, help='dot pitch px (default 6 * height/ref-height)')
    ap.add_argument('--legacy', action='store_true', help='exact old G11 reference look (6 px @1080p, no fade/min-dot/grain)')
    ap.add_argument('--seed', type=int, default=0, help='grain seed (use the frame index for moving grain)')
    ap.add_argument('--ref-height', type=float, default=1080, help='legacy only: height at which the pitch is 6 px'); ap.add_argument('--vignette', type=float, default=0.0)
    a = ap.parse_args(); img = cv2.imread(a.input, cv2.IMREAD_COLOR); h, w = img.shape[:2]
    if a.margin_only: panel = np.zeros((h, w), np.uint8)
    elif a.panel_mask: panel = cv2.imread(a.panel_mask, 0)
    elif a.panel: panel = tuple(int(v) for v in a.panel.split(','))
    else: panel = None
    acc = cv2.imread(a.accent, 0) if a.accent else None
    out = manga_bw(img, panel, acc, a.pitch, a.ref_height, a.vignette, a.legacy, a.seed)
    if a.protect: k = cv2.imread(a.protect, 0).astype(np.float32)[..., None] / 255.0; out = (out * (1 - k) + img * k).astype(np.uint8)
    cv2.imwrite(a.output, out)

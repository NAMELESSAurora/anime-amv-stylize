# -*- coding: utf-8 -*-
"""
电影质感收尾层（v10，用户 10-20："在 break 时播放羽蛇神和小贝的片段时可以将画面动态调整为 21:9，并且去掉两个 break 片段的点状纹理，
改为电影质感滤镜（增强特效和质感）"）。在整帧（镜头 + HUD + 转场 + 全局 glitch）之后、歌词 / 字幕之前做：
  · 胶片调性：温和 S 曲线（smoothstep 混 35%）+ 黑位轻抬（不是死黑）+ 高光肩部压缩
  · 辉光（受控 bloom）：亮部（亮度 > 0.62）在 1/4 分辨率上两档高斯（≈20 / 70 px）叠加，屏幕混合 —— 光效、火焰、光束更"亮出来"
  · 横向光条（变形宽银幕镜头的拉丝）：同一份亮部只做水平长模糊，冷白色，很淡
  · 边缘色散：R / B 通道绕画面中心各缩放 ±0.16%（角落 ~2 px，中心为 0）
  · 暗角：角落压 16%
  · 胶片颗粒：每帧换的细颗粒（半分辨率高斯噪声 → 轻模糊 → 双线性放大），中间调最强、暗部 / 高光弱；不是网点
  · 21:9 遮幅：上下黑边按 bars（0..1）收拢 / 打开（2560×1440 → 画面高 1097 px，每边 171 px）
所有项按强度 k（0..1）混合，进出 break 时跟遮幅一起渐变。
"""
import math
import numpy as np
import cv2
from style_frames import W, H, R

BAR_PX = (H - W * 9.0 / 21.0) / 2.0
_VIG = None


def _vig():
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / math.sqrt(2)
        _VIG = (r ** 2.2)[..., None]
    return _VIG


def _scale_ch(ch, s):
    M = np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]])
    return cv2.warpAffine(ch, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def finish(img, F, k=1.0, bars=0.0):
    if k <= 0.001 and bars <= 0.001:
        return img
    x = img.astype(np.float32) / 255.0
    if k > 0.001:
        s = x * x * (3 - 2 * x)                             # 调性：S 曲线 + 黑位轻抬 + 高光肩
        y = x + (s - x) * 0.35 * k
        y = y * (1 - 0.03 * k) + 0.022 * k
        y = np.where(y > 0.85, 0.85 + (y - 0.85) * (1 - 0.35 * k), y)
        sm = cv2.resize(x, (W // 4, H // 4), interpolation=cv2.INTER_AREA)   # 辉光 + 横向光条（取原图亮部）
        lum = sm.mean(2, keepdims=True)
        hi = sm * np.clip((lum - 0.62) / 0.38, 0, 1) ** 1.5
        b = 0.6 * cv2.GaussianBlur(hi, (0, 0), 5 * R / 1.333) + 0.4 * cv2.GaussianBlur(hi, (0, 0), 18 * R / 1.333)
        st = cv2.blur(hi.mean(2), (int(90 * R / 1.333) | 1, 1))[..., None] * np.array([0.80, 0.88, 1.0], np.float32)
        glow = cv2.resize(0.38 * b + 0.22 * st, (W, H), interpolation=cv2.INTER_LINEAR) * k
        y = 1 - (1 - np.clip(y, 0, 1)) * (1 - np.clip(glow, 0, 1))
        ca = 0.0016 * k                                     # 边缘色散
        y[..., 0] = _scale_ch(np.ascontiguousarray(y[..., 0]), 1 + ca)
        y[..., 2] = _scale_ch(np.ascontiguousarray(y[..., 2]), 1 - ca)
        y = y * (1 - 0.16 * k * _vig())                     # 暗角
        rng = np.random.default_rng(int(F) * 7919 + 17)     # 颗粒：每帧换
        g = rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
        g = cv2.resize(cv2.GaussianBlur(g, (0, 0), 0.6), (W, H), interpolation=cv2.INTER_LINEAR)
        l = y.mean(2, keepdims=True)
        amp = 0.030 * k * (0.35 + 0.65 * 4 * l * (1 - l))
        y = y + g[..., None] * amp
        x = y
    out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    bh = int(round(BAR_PX * bars))
    if bh > 0:
        out[:bh] = (5, 5, 6); out[H - bh:] = (5, 5, 6)
    return out


def ramp(t, w):
    """w = {t0, t1, bar_in: [a, b], bar_out: [a, b]}（成片秒）→ 当前强度 0..1（进 / 出都是 ease-in-out）。"""
    a0, a1 = w["bar_in"]; b0, b1 = w["bar_out"]
    if t < a0 or t > b1:
        return 0.0
    if t < a1:
        u = (t - a0) / (a1 - a0)
    elif t > b0:
        u = 1 - (t - b0) / (b1 - b0)
    else:
        return 1.0
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)

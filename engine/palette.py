# -*- coding: utf-8 -*-
"""
风格化配色（用户 10-19："少用橙色配色，你要根据人物配色来选择风格化配色"）。
每个镜头的强调色组（main / hi / dk / sec / txt）在 styled() 一开头统一换成"人物配色"：
  · 认得出的人物：角色签名色（主 / 副）；和画面主色色相接近（≤ 60°）时混 38% 画面色，冲突时只用角色色（不混出脏灰）
  · 认不出人物：仍跟画面走，但主色落在橙色区（15–48°）时换成画面副色；副色也是橙就用冷钢蓝
所有风格层（HUD / 档案页 / 漫画强调格 / 歌词荧光条 / 转场）都吃这一组颜色。
"""
import colorsys, re
import numpy as np

# 角色签名色（与 hud_v5.SIG 同一套）+ 配角
SIG = {"jeanne": ((226, 190, 96), (70, 96, 170)), "artoria": ((70, 120, 230), (232, 196, 90)), "artoria_alter": ((214, 44, 52), (150, 140, 170)),
       "okita": ((80, 204, 224), (244, 150, 186)), "okita_alter": ((226, 44, 66), (196, 186, 206)), "musashi": ((76, 136, 236), (226, 66, 76)),
       "medusa": ((160, 96, 226), (226, 80, 170)), "karna": ((244, 184, 64), (210, 60, 44)), "mordred": ((226, 56, 44), (206, 206, 216)),
       "mash": ((176, 126, 236), (244, 166, 206)), "richard": ((216, 56, 56), (244, 194, 84)),
       "ishtar": ((222, 52, 72), (240, 200, 92)), "arash": ((92, 150, 206), (226, 206, 150)), "bedivere": ((186, 200, 224), (84, 124, 204)),
       "lion_king": ((232, 204, 126), (232, 232, 244)), "quetzalcoatl": ((72, 196, 150), (240, 206, 96)),
       "hippolyta": ((206, 54, 80), (132, 86, 214))}
CAST = [("Saber Alter", "artoria_alter"), ("冲田总司〔Alter〕|冲田〔Alter〕", "okita_alter"), ("冲田总司", "okita"), ("宫本武藏", "musashi"),
        ("阿尔托莉雅|^Saber|着甲Saber", "artoria"), ("贞德", "jeanne"), ("^Rider|美杜莎", "medusa"), ("迦尔纳", "karna"), ("莫德雷德", "mordred"),
        ("玛修", "mash"), ("金发持剑青年|红披风剑士|金发红披风|理查", "richard"),
        ("卫宫士郎|士郎", ((206, 52, 64), (110, 200, 214))), ("远坂凛|凛", ((214, 40, 60), (60, 60, 90))),
        ("黑樱|间桐樱|樱", ((176, 40, 96), (126, 70, 170))), ("Archer|エミヤ", ((200, 44, 44), (196, 200, 212))),
        ("言峰", ((124, 76, 166), (70, 60, 80))), ("伊莉雅", ((222, 222, 240), (194, 84, 146))), ("齐格", ((96, 200, 140), (220, 220, 232))),
        ("塞米拉米斯", ((64, 166, 104), (52, 48, 60))), ("恩奇都", ((92, 204, 144), (232, 232, 222))), ("提亚马特", ((152, 100, 222), (84, 202, 222))),
        ("贝德维尔", "bedivere"), ("狮子王", "lion_king"), ("伊什塔尔", "ishtar"), ("阿拉什", "arash"), ("魁札尔", "quetzalcoatl"), ("希波吕忒", "hippolyta"),
        ("尼禄", ((222, 42, 52), (240, 200, 92))), ("藤丸|主人公|黑发少年|男主人公", ((84, 124, 224), (232, 232, 244))),
        ("库·丘林|Lancer", ((62, 112, 222), (222, 62, 62))), ("Berserker|赫拉克勒斯", ((150, 140, 140), (222, 62, 52))),
        ("阿喀琉斯", ((84, 204, 124), (232, 192, 92))), ("阿斯托尔福", ((242, 152, 204), (232, 232, 244))),
        ("Assassin|哈桑|山之翁|骷髅面具", ((84, 96, 140), (204, 212, 232))), ("天草", ((206, 206, 214), (200, 60, 70))),
        ("狮子劫", ((110, 120, 90), (210, 70, 60))), ("阿塔兰忒", ((96, 190, 110), (232, 200, 120)))]
STEEL = (110, 168, 222)
_K = {}


def _hsv(c):
    return colorsys.rgb_to_hsv(*(np.clip(np.asarray(c, np.float32), 0, 255) / 255.0))


def _rgb(h, s, v):
    return tuple(int(round(x * 255)) for x in colorsys.hsv_to_rgb(h % 1.0, min(1.0, max(0.0, s)), min(1.0, max(0.0, v))))


def colors_of(who):
    """人物配色 (main, sec) 或 None；按 who 字段的第一个名字匹配。"""
    if who in _K:
        return _K[who]
    first = re.split(r"[、,，；;/+]", who or "")[0].strip()
    out = None
    for pat, v in CAST:
        if re.search(pat, first):
            out = SIG[v] if isinstance(v, str) else v
            break
    _K[who] = out
    return out


def is_orange(rgb):
    h, s, v = _hsv(rgb)
    return 15 / 360 <= h <= 48 / 360 and s > 0.35


def build(main, sec):
    h, s, v = _hsv(main)
    S = float(np.clip(s, 0.50, 0.90)) if s > 0.12 else s            # 白 / 银这类低饱和角色色保持低饱和
    mn = _rgb(h, S, max(v, 0.86))
    hs, ss, vs = _hsv(sec)
    y = 0.299 * mn[0] + 0.587 * mn[1] + 0.114 * mn[2]
    return dict(main=mn, hi=_rgb(h, S * 0.42, 1.0), dk=_rgb(h, min(1.0, S * 1.05), 0.40), sec=_rgb(hs, max(ss, 0.25) if ss > 0.12 else ss, max(vs, 0.85)),
                txt=(14, 12, 14) if y > 150 else (246, 242, 236), hue=h * 360, hue2=hs * 360)


def shot_acc(who, acc):
    cc = colors_of(who)
    fm = acc["main"]
    if cc is not None:
        hm = _hsv(cc[0])[0] * 360; hf = _hsv(fm)[0] * 360
        dh = min(abs(hm - hf), 360 - abs(hm - hf))
        main = (0.62 * np.asarray(cc[0], np.float32) + 0.38 * np.asarray(fm, np.float32)) if dh <= 60 else np.asarray(cc[0], np.float32)
        return build(main, cc[1])
    if is_orange(fm):
        sec = acc.get("sec", STEEL)
        return build(sec if not is_orange(sec) else STEEL, fm if not is_orange(fm) else (200, 206, 214))
    return acc

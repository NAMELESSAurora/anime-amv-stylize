# -*- coding: utf-8 -*-
"""每个英灵一套灵基卡框（用户 10-19："card frame 你可以自己设计一套适合英灵的"）。用户提供的图像 API，透明底；产物 assets/gen/frame_<key>.png。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from imgapi import generate

BASE = ("An ornate vertical trading-card frame for {who}. {look} Intricate, precise, high-end metalwork border, perfectly symmetrical left-right, "
        "rich detail concentrated on the border and the top crest. The large central window is completely empty and fully transparent. "
        "No plaques, no banners with blank space, no text, no letters, no numbers, no people, no faces. Front view, crisp clean edges, "
        "isolated on a fully transparent background.")
FR = {
    "jeanne": ("a holy saint-knight", "Silver and pale gold with deep navy stained-glass inlays, fleur-de-lis ornaments, white lilies, a flowing banner motif and fine chains along the sides."),
    "artoria": ("the King of Knights", "Polished gold with royal-blue enamel, a sword standing upright as the top crest, Celtic knotwork, apple-blossom and lake-ripple motifs."),
    "artoria_alter": ("a corrupted black king", "Blackened steel with thin glowing crimson veins, jagged thorny gothic ornaments, a dark sword as the top crest."),
    "okita": ("a Shinsengumi swordswoman", "The left and right borders are folded light-blue asagi haori fabric with the bold white dandara mountain stripes; sakura branches on the corners; a katana in a black scabbard across the top as the crest and a second blade along the bottom."),   # v10 用户："蓝色羽织+樱花以及武士刀"（现用 gen_frame_okita_v2.py 的 b 版）
    "okita_alter": ("a dark demon swordswoman", "Glossy black lacquer and crimson, a crescent moon top crest, red sakura petals, thin void-black cracks with red light."),
    "musashi": ("a wandering two-sword samurai", "Indigo blue and vermilion with gold leaf, two crossed katana as the top crest, stylised ocean waves, a five-rings motif."),
    "medusa": ("a gorgon rider", "Deep violet and silver, serpent scales, intertwined snakes along the sides, a winged pegasus top crest, ancient Greek meander pattern."),
    "karna": ("the son of the sun god", "Radiant gold armour plates, a sun-disc top crest with flame rays, crimson jewels, Indian temple filigree."),
    "mordred": ("a rebel knight", "Crimson and silver steel, red lightning motifs, a horned helmet top crest, a broken crown, fierce angular ornaments."),
    "mash": ("a shield-bearing knight", "Lavender and silver, a large round shield with a cross as the top crest, a round-table circle motif, soft feather details."),
    "richard": ("the Lionheart crusader king", "Crimson and gold, a lion head top crest, a crusader cross, small heraldic shields and sword motifs."),
}
only = sys.argv[1:] or list(FR)
for key in only:
    who, look = FR[key]
    out = os.path.join(r"D:\A\provant\assets\gen", "frame_%s.png" % key)
    if os.path.exists(out):
        continue
    for k in range(10):
        try:
            p, meta = generate(BASE.format(who=who, look=look), out, model="gpt-image-2.5-flare" if k % 2 == 0 else "gpt-image-2.5-sunburst", size="1024x1536")
            print(key, "ok", meta["seconds"], "s", flush=True); break
        except Exception as e:
            print(key, "try", k, str(e)[:120], flush=True); time.sleep(15 + 15 * k)

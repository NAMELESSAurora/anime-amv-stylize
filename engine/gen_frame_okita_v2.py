# -*- coding: utf-8 -*-
"""冲田总司卡框 v2（用户 10-20："冲田总司的边框应该是蓝色羽织+樱花以及武士刀"）：两个模型各出一张，挑一张替换 frame_okita.png。"""
import os, sys, time, threading
sys.path.insert(0, os.path.dirname(__file__))
from imgapi import generate
from gen_frames_v4 import BASE

LOOK = ("The left and right borders are made of folded light-blue asagi haori fabric with the bold white jagged mountain-stripe dandara "
        "pattern along the hems, clearly visible as the Shinsengumi haori. Branches of pale pink sakura cherry blossoms wrap the four corners, "
        "with a few drifting petals. A katana in a black lacquer scabbard with silver fittings and a round tsuba lies horizontally across the "
        "top as the crest, and a second slender katana blade runs along the bottom edge. Silver metal trim.")


def run(tag, model):
    out = os.path.join(r"D:\A\provant\assets\gen", "frame_okita_v2%s.png" % tag)
    for k in range(8):
        try:
            p, meta = generate(BASE.format(who="a Shinsengumi swordswoman", look=LOOK), out, model=model, size="1024x1536")
            print(tag, "ok", meta["seconds"], "s", flush=True); return
        except Exception as e:
            print(tag, "try", k, str(e)[:120], flush=True); time.sleep(15 + 15 * k)


ts = [threading.Thread(target=run, args=a) for a in (("a", "gpt-image-2.5-flare"), ("b", "gpt-image-2.5-sunburst"))]
for t in ts:
    t.start()
for t in ts:
    t.join()

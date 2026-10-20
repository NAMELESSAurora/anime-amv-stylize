# -*- coding: utf-8 -*-
"""
透明底素材图生成（用户 10-19 提供的图像 API，newapi 网关，OpenAI 兼容 /v1/images/generations）。
凭据在 ~/.provant_imgapi.json（私人：不写进工程、不打印、不打包）。
用法: python scripts/imgapi.py <out.png> "<prompt>" [--model M] [--size WxH] [--opaque]
产物放 assets/gen/，旁边写同名 .json 记录 prompt / 模型 / 尺寸（不含凭据）。
"""
import base64, json, os, sys, time, urllib.request, urllib.error

CFG = json.load(open(os.path.join(os.path.expanduser("~"), ".provant_imgapi.json"), encoding="utf-8"))


def generate(prompt, out, model="gpt-image-2.5-flare", size="1024x1024", transparent=True, quality="high", timeout=600):
    body = {"model": model, "prompt": prompt, "size": size}            # 只发网关认的字段（多余字段会被拒成 model_temporarily_unavailable）
    if transparent:
        body["background"] = "transparent"
    req = urllib.request.Request(CFG["base"].rstrip("/") + "/v1/images/generations", data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": "Bearer " + CFG["key"], "Content-Type": "application/json"})
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP %d: %s" % (e.code, e.read().decode("utf-8", "replace")[:600]))
    item = d["data"][0]
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
    else:
        raw = urllib.request.urlopen(item["url"], timeout=timeout).read()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "wb").write(raw)
    meta = {"prompt": prompt, "model": model, "size": size, "transparent": transparent, "quality": quality, "seconds": round(time.time() - t, 1),
            "revised_prompt": item.get("revised_prompt")}
    json.dump(meta, open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out, meta


if __name__ == "__main__":
    a = sys.argv[1:]
    model = a[a.index("--model") + 1] if "--model" in a else "gpt-image-2.5-flare"
    size = a[a.index("--size") + 1] if "--size" in a else "1024x1024"
    out, meta = generate(a[1], a[0], model=model, size=size, transparent="--opaque" not in a)
    from PIL import Image
    im = Image.open(out)
    print(out, im.size, im.mode, meta["seconds"], "s")

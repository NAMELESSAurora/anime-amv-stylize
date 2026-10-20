# 10 字体与素材

## 字体（assets/fonts，均从 github.com/google/fonts 下载）
| 文件 | 许可 | 用途 |
|---|---|---|
| SairaExtraCondensed-Black / -Bold | OFL | HUD DISPLAY：巨型描边字、等级字母、卡面职阶 |
| Rajdhani-Bold / SemiBold / Medium | OFL | HUD H1 / H2：真名、数值 |
| ChakraPetch-Medium / Bold | OFL | HUD LABEL：全大写标签 |
| ShareTechMono-Regular | OFL | HUD MICRO：微型数据 |
| Bangers-Regular | OFL | 美漫旧版标题字 |
| Bungee-Regular / BungeeShade | OFL | 美漫拟声字、职阶小标 |
| ComicNeue-Bold | OFL | 美漫说明框手写字 |
| Knewave / RubikMonoOne | OFL | 备用 |
| PermanentMarker | Apache 2.0 | 备用 |
系统字体：Bodoni MT Black（海报 / slit 强调字）、Noto Serif SC VF（明朝：说明框、宝具名）、思源黑 VF（日文名）、Michroma（旧 HUD）。
下载：`curl -sL -o X.ttf https://github.com/google/fonts/raw/main/ofl/<family>/<File>.ttf`（Permanent Marker 在 apache/）。

## 角色资料（servant_info.json）
让子代理逐个抓 FGO Wiki / TYPE-MOON Wiki（HTML 被 Cloudflare 挡时走 api.php），再用 Atlas Academy JP 游戏数据（api.atlasacademy.io/nice/JP/servant/<id>?lore=true）交叉核对；每条带来源；查不到填 null，不猜。注意多版本（泳装 / 不同作品的参数块不同）、日文写法（〔オルタ〕、遙 / 遥、Ⅰ 是罗马数字字符）。

## 透明底素材生成（用户提供的图像 API）
- 客户端 `engine/imgapi.py`；凭据在 `~/.provant_imgapi.json`（{"base":…, "key":…}，私人，不进工程 / 不打印 / 不打包）。
- 网关 = OpenAI 兼容 `/v1/images/generations`。**只发 model / prompt / size / background="transparent"**；多发字段（quality、output_format、n）会被拒成 `model_temporarily_unavailable`。
- 实测：1024x1536 请求返回 848×1264 RGBA；2048x2048 可以；3840x2160 当时 429。常见 429 / 503 / 超时 → 退避重试、两个模型轮换；单张 20–450 s，后台跑。
- 提示词要点：`isolated on a fully transparent background`、`no text, no letters, no numbers, no people, no faces`、对称、正视图；卡框要写明"中央大窗口完全透明，不要空白铭牌"（不然会生成空牌子，上面的字只能自己画）。
- 用法原则：生成素材**不直接展示**——先风格化（配色映射、网点、扫描线、glitch），再按画面合成；贴图式叠加（法阵）容易"像两个图层"。

### 卡框提示词模板（gen_frames_v4.py）
```
An ornate vertical trading-card frame for {角色气质}. {材质 / 配色 / 母题 / 顶饰}. Intricate, precise, high-end metalwork border,
perfectly symmetrical left-right, rich detail concentrated on the border and the top crest. The large central window is completely
empty and fully transparent. No plaques, no banners with blank space, no text, no letters, no numbers, no people, no faces.
Front view, crisp clean edges, isolated on a fully transparent background.
```
例：贞德 = 银 + 淡金 + 深蓝彩窗、鸢尾花、百合、旗帜、链饰；冲田 = 浅青白漆 + 银、新选组段染（だんだら）、樱、刀镡顶饰；武藏 = 靛蓝 + 朱 + 金箔、双刀交叉、浪纹、五轮。

## 中央窗口检测
alpha < 24 的连通域里包含画面中心的那块 → 外接框 = 放画面的窗口（`v4_looks._frame`）。

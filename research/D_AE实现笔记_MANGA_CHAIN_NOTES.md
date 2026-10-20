# 漫画风格链 — 实现笔记

记录 v7 → v8 → v10 的演进，以及 AE ExtendScript 上踩过的坑。
**这些东西不看文档重来一遍会再踩，所以写下来。**

---

## 一、图层时间语义（踩了三次，最贵的坑）

AE 图层的时间是三个互相牵制的属性，理解错就会出现"时长变 0"：

| 属性 | 含义 |
|---|---|
| `startTime` | 这个图层的**源**第 0 秒，落在合成时间轴的哪一刻 |
| `inPoint` | 图层在合成时间轴上**开始出现**的时刻 |
| `outPoint` | 图层在合成时间轴上**结束**的时刻 |

正确的做法是：**`startTime` 和 `inPoint` 设成同一个值**，让源的 0 秒对齐到该出现的位置。

```javascript
// 对：源从第 0 秒开始播，出现在 S.t0
pl.startTime = S.t0;
pl.inPoint   = S.t0;
pl.outPoint  = S.t1;
```

```javascript
// 错：这会让 AE 去取「源的第 39.59 秒」，而源根本没这么长，
//     outPoint 被钳到 inPoint，时长变成 0
pl.startTime = 0;
pl.inPoint   = S.t0;
pl.outPoint  = S.t1;
```

**另外两条约束：**

1. **合成时长必须 ≥ 任何图层的 outPoint。** 否则所有图层的 outPoint 都被钳到合成时长。
2. **预合成/素材的 `source.duration` 必须 ≥ 它在合成上要占用的跨度。** 放在 39.59s 处的图层，
   如果源只有 25 秒，`startTime + source.duration` 就超过主合成时长，又回到钳制问题。

调试手法：读回 `startTime / inPoint / outPoint / source.duration` 四件套，一眼能看出是哪条约束卡住。

---

## 二、ExtendScript 的 ES3 限制

- **没有 `Array.prototype.forEach`**，`[a,b].forEach(...)` 直接报 `forEach is undefined`。用 `for` 循环。
- 没有箭头函数、`let` / `const`、模板字符串。
- `.jsx` 被 `$.evalFile` 加载时，**顶层不能有 `return`**（"Illegal 'return' outside of a function body"）。

## 三、关键帧缓动

`KeyframeEase(speed, influence)` 的 **influence 合法区间是 0.1–100**，给 0 会报
`Unable to set "influence". Value 0 out of range 0.1 to 100`。想要"两端平滑"就用 33.33（AE 默认）。

```
setTemporalEaseAtKey(keyIndex, inEaseArray, outEaseArray)
```
数组长度必须等于该属性的**维度数**（位置是 2 或 3，缩放是 3）。

## 四、效果参数与陷阱

- **`ADBE AutoLevels` 的参数名带单位**：`Temporal Smoothing (seconds)`，不是 `Temporal Smoothing`。
- **Pro Levels2 的值是 0–1 归一化**（16bpc 工程里也是）。
- `ADBE Posterize` 的参数名是 **`Level`**（单数）。
- **`ADBE Levels2`（旧版 Levels）在这台机器上不存在**，用 `ADBE Pro Levels2`。
- **Color Halftone 效果缺失**，所以网点是用形状图层的椭圆 + 两个 Repeater 手工搭的。
- 给 `ADBE Effect Parade` **加完所有效果再取子属性**。边加边取的话，之前拿到的引用会失效
  （"ReferenceError: Object is invalid"）。
- `fx()` 只返回同 matchName 的**第一个**效果；一个图层上有两个同类效果时要用 `fxNth(layer, matchName, n)`。

---

## 五、风格链本体：v7 → v8

### v7（单帧好看，多格失效）

结构：BASE 只出黑白两极（`Posterize 3` + gamma 1.60），中间调交给网点层。

**问题**：BASE 的 `Input White = 0.46` 是按单帧调出来的常数。放进多格分镜后：
- 暗场（源 81% 像素在最暗档）→ 整格糊成纯黑
- 亮场（烟尘、云）→ 大面积冲成纯白，中间调消失

根因：**一个固定阈值不可能同时适配所有镜头。**

### v8（当前）

每层前面加 **Auto Levels**，先把该镜头自己的动态范围拉满，再做二值化：

```
BASE    : AutoLevels → Black&White → Pro Levels2(IW 0.85, γ1.5)
          → Posterize(4) → Pro Levels2(γ1.30)
BASE_MTE: AutoLevels → Black&White → Pro Levels2(IB 0.18, IW 0.88)
LINES   : AutoLevels → Black&White → Find Edges → Pro Levels2(IB 0.30, IW 0.85)
DOTS    : 形状层，椭圆 + 2×Repeater，不透明度 88，LUMA_INVERTED 遮罩
GRAIN   : 白固态 + Noise2(12)，OVERLAY，不透明度 5
```

`AutoLevels` 的 `Temporal Smoothing (seconds) = 0.5`：逐帧重算黑白点会让画面"沸腾"，
0.5 秒的平滑压住它。**但这也意味着极快的闪白/闪黑会被平滑掉**——如果某段需要这个，
要么调小平滑，要么那段不走 AutoLevels。

网点尺寸跟着格子走：`PITCH = (pw < 700) ? 17 : 21`。

### 关键结构（不要改）

**网点层和颗粒层是独立于画面的**，所以格内推拉时，**网纹固定在纸面上、画面在屏幕底下移动**。
这是实体漫画印刷的观感。如果图带着网点一起缩放，就变成"一张图在动"，不是漫画。

**格线压在画面上**（层序：FRAME → GRID → 各格画面），所以画面从格线底下穿过。
这是正确的漫画层序。破格元素要放在格线之上。

---

## 六、格内运镜

设计文档 179 行：**推拉 ≤ 8%，位移 ≤ 6%**。

正确实现有两个容易错的点：

1. **位移按格子尺寸算，不是按源尺寸算。** 源是 1920 宽、格子是 948 宽，
   用 `W * 0.03` 得到的是格子宽度的 6.1%，单镜头来回 12%——超上限两倍。应该用 `pw * 0.03`。
2. **基准缩放要先留出运镜余量。** 基准正好等于 cover 时，位置一偏移就在格子边上露白。
   最坏情况需满足 `s * sourceSize / 2 >= panelSize / 2 + 位移`：

```javascript
var sCover = Math.max(pw / W, ph / H) * 100;
var dX = pw * 0.03, dY = ph * 0.03;      // 格子尺度的位移
var s = Math.max(sCover,
                 (pw + 2 * dX) / W * 100,   // 横向够不够
                 (ph + 2 * dY) / H * 100);  // 纵向够不够
```

校验方法：算出图像四边在格子坐标系里的位置，**余量必须全部 ≤ 0**（负数 = 超出格子 = 有富余）。

---

## 七、选镜头的约束（实测得出）

漫画处理会把**暗场变成大面积墨黑**。这在漫画里合法（纯黑本来就是墨），
但格子里需要观众"读出内容"时就不能挑这种镜头。

实测 `fKoHcAM0PEk` @21.74s：**81.1% 的像素落在 0–25 亮度档，平均亮度 19.2/255**。
Auto Levels 已经把这帧仅有的动态范围拉到极限，剩下的黑是源里真的没有信息。

判断方法：抽原帧算亮度直方图。若 >60% 像素在最暗两档，这个镜头进漫画链会变墨块。

---

## 八、v10 渲染实测与 Gamma 修正（2026-10-17）

`out/sample_E05_v10.mp4` 逐格采样（每 0.75s，共 76 个格子样本，内缩 6px 避开格线）：

| 指标 | γ1.0（v10） |
|---|---|
| 墨黑（亮度 <40）平均 | 51.8% |
| 纸白（>215）平均 | 15.3% |
| 墨黑 >70% 的格子 | 31 / 76 |

结论：画面一半是墨，暗场镜头读不出内容。BASE 第一级 Pro Levels2 的 Gamma 改为 **1.5**
（在 Posterize 之前提亮中间调，暗部落进网点档而不是直接变墨）。

**已知副作用**：Auto Levels + γ1.5 会放大源的压缩块（8×8 马赛克）和暗部噪点，
夜景天空边缘能看到方块。这是源质量问题，漫画化会把它暴露出来——选镜头时要避开压缩严重的暗场。

## 九、渲染：用 aerender，不用 MCP 的 ae_render

2026-10-17：`ae_render` 在 GUI 里 30 分钟无进度回报，被 MCP 超时中止，零输出。
改用 `scripts/render_guard.py`（aerender + 看门狗）：451 帧 2 分 10 秒，峰值 13.4 GB。

- aerender 读**磁盘上的 .aep**，渲染前必须先保存工程。
- 看门狗只杀 aerender 自己的进程树（`taskkill /T /PID`），不会动用户开着的 AE 界面。
  nostalgia-demo 那份旧脚本按名字匹配 `AfterFX*`，会连 GUI 一起杀，不要照搬。
- 音频不走 aerender，用 `scripts/mux_sample.py` 从原 WAV 按帧号切（起点 = 起始帧 / 24）后合成 mp4。
- 渲染队列里残留的旧任务要先清掉，否则 aerender 不受影响，但 GUI 里再点渲染会重复跑。

## 十、Pixy Halftone 插件实测（2026-10-17）

matchName `ADBE Esso_Halftone`（分类 Animation Gym）。依赖 VS 调试运行库（VCRUNTIME140D / ucrtbased），
已装 VS Build Tools 2022 解决；装之前 AE 启动会报错、插件加载失败。

参数：Shape 1–6、Color（RGBA）、Size 1–1000 px、Space 0–500、Contrast −255–255、Roughen 1–255、
Invert、Alpha、Reveal Original、Pixlate。Shape 实测（`qa/pixy_ramp.png`，Size 24）：
1 方点 / 2 圆点 / 3 三角 / 4 菱形 / 5 斜线（万線）/ 6 混合形状。

**灰底的根因**（`qa/pixy_ramp_v2.png` 实测，不是 Color 参数也不是混合模式）：
- Pixy 的点是**半透明**的：点的 alpha 随暗度线性变化（亮度 0.12 时 alpha 29/255），点径也随暗度变化。
  叠在纸白上就成了"灰色的点"，所以中间调发灰、发虚。
- 修法：在 Pixy 后面加 Pro Levels2，**Alpha Input White 0.15**，把点的 alpha 拉满成实心墨。
- 拉满之后墨量仍不是 1−L：L=0.3 时墨 28%、L=0.5 时 13%、L=0.7 时 6%（理想 70/50/30%）。
  Contrast +60 加深暗部（L=0.3 → 42%），但中间调和亮部不变。
- 结论：Pixy 适合**做效果层**（波普粗网、三角/菱形装饰网、局部网点），不适合当主网点引擎；
  主网点继续用自制网屏阈值合成（线性误差 ≤ 5%，见第五节和 79_CAL_SCREEN）。

## 十一、G11 风格（grokbot 风格包移植，2026-10-17）

来源 `D:\备份\manga_bw_g11_style.zip`（README + numpy/opencv 脚本 + 参考图），解包在 `research/grokbot/`。
AE 版：`build_g11.jsx` → `73_G11_<name>`（MARGIN 74_ / PANEL 75_ 两个预合成 + 格子遮罩 + 5 px 框）。

- 格内：AutoLevels → 黑白 → 对比 1.1（Levels 0.045/0.955）→ 单色颗粒。**不打网点、不二值化**。
- 格外：画面亮度（模糊 2 px）+ 径向渐变 SCREEN（中心白、Input White 0.61 ≈ r 0.45 起渐强）
  → 自制网屏 `screen_dot_p14_a45.png` 阈值合成。网点固定在纸面上。
- 强调色：Leave Color 按 **RGB** 匹配（Match colors = 1）+ Extract 只留亮部，COLOR 混合盖回灰阶。
  不能按色相：S033 的有色像素 83% 在 0–60°，按色相会整格回色。
- 坑：**Add Grain 的 Viewing Mode 默认是 Preview**，只在画面中间一个框里出颗粒并画白框；必须设 3（Final Output）。
- 与 grokbot 原版的差异：网点间距 14.14 px（原版 10 px@1080p，低于本项目 12 px 下限）；
  强调色自动选取（原版是手绘瞳孔遮罩）。
- 它的"不要做"清单（格内不分档、不大面积纯黑）与 M 系列相反——G11 是**另一种风格**，不替换 M 系列。

## 十二、$.evalFile 的返回值

`ae_run_script` 的脚本体会被包进函数执行，所以 `$.evalFile(...); return __GLOBAL;` 可以正常返回；
**被 evalFile 的 .jsx 文件里**不能有顶层 `return`（报 "Illegal 'return' outside of a function body"）。
约定：jsx 把结果写进全局变量（`__LAB` / `__G11` / `__SAMPLE`），调用方读这个变量。

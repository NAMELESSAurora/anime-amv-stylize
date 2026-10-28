# 11 · ER 后处理层（`engine/v8_post.py`）

来自 Edgerunners AMV（v5 系列）在这套引擎上的第二个项目。把那边验证过、用户点头的画面后处理整理成**通用函数**：只吃 RGB 数组 + 脸框（像素 x0,y0,x1,y1），分辨率无关，不含任何工程路径、时间点、歌词、台词或素材。

**向后兼容**：`v2_preview` 不 import 它，旧工程行为不变。要用就在自己的 `styled()` / `render_frame()` 包装里调用（和 v5_tape / v7_cine 的接法一样）。时间点全部来自你的 EDL。

自检：`python engine/v8_post.py` → `v8_post smoke OK`（640×360 合成图，几秒，不需要 GPU / 素材）。

## 全局开关 `CFG`
| 键 | 默认 | 说明 |
|---|---|---|
| `face_cap` | 225 | QC 硬线：编码后每个脸框 p95 亮度 ≤225 |
| `face_cap_inner` | 220 | face_guard 压到 220，给 4:2:0 编码的饱和红留余量 |
| `speed_lines` | **False** | 放射速度线默认关（用户多次否决）；要用 `CFG["speed_lines"]=True` 或 `force=True` |
| `grain` | 3.0 | 细颗粒 σ，永远不上脸 |
| `halftone_cell` | 10 | 网点格 |
| `max_flash_per_s` / `min_flash_gap_s` | 3 / 0.34 | 同 08_qc.md |

## 函数速查
| 分组 | 函数 | 要点 |
|---|---|---|
| 脸保护 | `face_mask` `face_guard` `qc_face` | **最后一步**（歌词 / HUD 之后）调用 face_guard；脸上不加颗粒、网点、速度线、glitch 边 |
| 调色 | `grade(kind)`：cool / cool_l / red / purple / teal / split / **d2** | d2 = 夜城 look（暗部青、高光暖中性、轻 S 曲线、保线条）；split = 对角蓝 / 红 + 细亮缝 |
| 色块跳 | `color_jump(kinds, t, period)` | 每次跳色在 QC 里算一次闪 |
| 光效 | `light_glow` | 只在小高光上：局部变形镜头横向光条 + 点光；脸上只留镜面点；`burst` = 打点时加十字星芒 |
| 五条悟式 | `panel_stack` `grid_zoom` + `grid_zoom_curve` `diagonal_shards` `flash(invert/bw)` `halftone` `rings` | 叠画格逐拍落下 → 九宫格拉远 → 对角碎裂 → 反色 / 黑白闪 + 震动；速度线可选、默认关 |
| 冲击 | `impact_frame` `slam` | 双色墨稿冲击帧（脸部阈值放宽，不变成白板）+ 缩放砸入 + 衰减震动 |
| 叠化转场（v5g） | `ov_split` `ov_match` `ov_frames` | 斜切 / 主体框对齐的 match cut / 三格斜框依次滑出；**擦除边永远不切过脸**（脸框内覆盖率取均值） |
| 画中画（v5h） | `pip_place` `pip_stack` `pip_cut_gain` | 叠层小窗避开脸 + 发型区、左右交替；暖色跳不把脸推过 225；硬切到 B 后增益 0.88 再缓回 |
| 漫画 | `free_panels` `manga_page` `manga_transition` | 自由分格页：只缩小不放大（INTER_AREA）、1px 抗锯齿硬边、可选 duo；整帧缩进自己的格子（push / pull / panL） |
| HUD | `hud_fine` | 细线精密 HUD：角标、刻度尺、扫描标、小标签（只写核对过的信息，见 03_hud.md） |
| 剪辑 | `letterbox` `letterbox_join` `jcut` | 遮幅收 → 切 → 放，藏剪点且没有黑帧；J-cut 音频先入 |
| 歌词动效 | `LYRIC_MOTION` + `lyric_motion` | `hot`（v5a 关键词砸入 + 回弹 + 淡墙纸）/ `insert`（v5f 衬线斜体 + 大号幽灵字、避脸）/ `line`；只给变换，不含歌词 |
| QC | `qc_face` `qc_flashes` | 脸 p95；闪烁频率 / 间隔 / 黑帧 |

## 经验（来自 ER 审片）
- 高饱和红 duo 在 4:2:0 编码后脸会多出 3–5 级亮度 → 渲染时压 220，QC 按 225 验。
- 网点 / 颗粒 / 速度线上脸，用户一眼就否；所有纹理都乘 `(1 - face_mask)`。
- 光效只做"局部"：大面积 bloom 会糊线条（违反硬规则"不能化开"）。
- 漫画页用整帧原画裁格，不要放大小图；格子边是硬边，不羽化。
- 叠化转场的擦除边扫过脸会显得廉价 → `_face_safe_mix`。
- 内存：全分辨率 float32 很吃内存，多进程渲染时每 worker 预留 6–8 GB（见 08_qc.md 内存部分）。

# engine/ — PROVANT v7 时的代码快照（2026-10-19）

原工程：`D:\A\provant\scripts\`（那边是活代码；这里是可复用的拷贝）。全部 Python 3.12；GPU 部分在 `.venv_gpu`（system-site-packages + onnxruntime-directml + moderngl）。

## 模块
| 文件 | 作用 |
|---|---|
| `style_frames.py` | 常量（W/H/R、字体路径、ffmpeg 路径）、文字工具（squeeze_text / spaced / vfont）、灰阶 / 网点 / lineart |
| `cards.py` | LiveSource（ffmpeg 解码流）、src_path、缓动（eo / bo / cl）、halftone_c（缓存网点场） |
| `gpu.py` | moderngl 着色器：r_base、c_dark_base、over、compose、hud3（HUD 后景世界锁定 + 遮挡 + 调光 + 泛光 + 运动模糊 + 行 glitch） |
| `v2_looks.py` | 强调色轨、抠像、R / Cd / Cb / P 基础风格、波形撕裂、boil、星座、像素生长、裁切框、歌词旧样式 |
| `v2_trans.py` | glitch / slam / lock_zoom / panel_slap / whip / grid |
| `v3_hex.py` | EVA 蜂巢：hex_field、hex_wipe |
| `v3_manga.py` | 漫画页：纸面网点、四版式、tone（mono / duo / ink / color）、竖排说明框、速度线、riso 海报 |
| `v4_comic.py` | 美漫（保真版）+ 纵横宇宙式说明框 / 拟声字 |
| `v4_looks.py` | 灵基卡（卡框风格化 + glitch）、slit 转场 |
| `palette.py` | 角色签名色 + 画面色合成（每镜强调色） |
| `hud_elems.py` `hud_v2.py` `hud_v3.py` `hud_v4.py` | 旧版 HUD（Canvas 基础类在 hud_v2；info_for / WORK 在 hud_v4） |
| `hud_v5_kit.py` `hud_v5.py` | 现行 HUD：设计 token、Clock 动效、组件、6 个模板、公共精密层 |
| `v2_preview.py` | **引擎**：prep / analyze_shot / Ctx / styled / hud_on / render_frame / 分块渲染 / encode / sheet；`python v2_preview.py KEY analyze|render|encode|sheet|all` |
| `edl_v9.py` | 拍点槽位式 EDL 生成器（段落简报、选镜、look / HUD / 转场 / slit / 歌词 / 台词 / 闪烁预检 / 去重） |
| `edl_v10.py` | **最新 EDL 生成器（按角色分块）**：BLOCKS（连续源 / 锚点 / 慢放权重 / nohud）+ 复用 edl_v9 的 make_shot、HUD、转场、歌词、台词 |
| `timeline_flash.py` | 成片级闪烁预检（qc 阈值，报告违规窗和对应镜头） |
| `src_luma.py` | 单个池镜头的逐帧亮度 / 跳变位置（收 s0/s1 用） |
| `pool_v2.py` `build_shot_pool.py` `shot_inventory.py` `make_sheets.py` | 镜头切分、指标、审片表、镜头池 |
| `conform_footage.py` `detect_letterbox.py` | 素材规范化、黑边检测 |
| `adaptive_grid.py` `beat_verify.py` | 自适应拍网格 + 核验 |
| `lyric_time.py` | 逐词定时（干声起音 DP + MANUAL / END 覆盖） |
| `sep_mdx.py` | UVR MDX Voc_FT 人声分离 |
| `dialog_harvest.py` | 台词候选采集（分离 → VAD → F0 → Whisper） |
| `v2_audio.py` `v3_audio.py` | 原声事件 + 台词混音（EQ / 压缩 / M-S 挖槽 / 限制器）+ mux |
| `qc_v1.py` | 逐帧亮度 QC（黑帧 / 闪烁） |
| `anime_det.py` | deepghs YOLO 脸 / 头 / 人检测 |
| `imgapi.py` `gen_frames_v4.py` | 透明底素材生成（凭据在 ~/.provant_imgapi.json） |
| `data/servant_info.json` | 11 个 Fate 角色核对过的资料（带来源） |

## 外部依赖
ffmpeg（含 hevc_nvenc / h264_nvenc）、numpy、opencv-python、Pillow、scipy、librosa、soundfile、onnxruntime(-directml)、moderngl、faster-whisper。
模型：`models/anime-seg/isnetis.onnx`、`models/anime_face_detection/…`（deepghs YOLOv8）、`models/uvr/models/MDXNet/UVR-MDX-NET-Voc_FT.onnx`、`models/silero-vad/onnx/model.onnx`。

## 移植时要改的硬编码
| 位置 | 原值 | 改成 |
|---|---|---|
| `style_frames.py` ROOT / FF | `D:\A\provant`、`E:\NAMELESS\tools\ffmpeg-9.0.1\…\ffmpeg.exe` | 新工程根、本机 ffmpeg |
| `v2_preview.py` prep() 工作目录 | `E:\provant_work\<KEY>` | 新工作盘 |
| `v2_preview.py` encode / sheet 输出映射 | `{"V5":"v3",…,"V9":"v7"}` | 新 KEY → 版本号 |
| `v2_preview.py` encode / `v2_audio.py` / `v3_audio.py` 原曲 | `D:\备份\*PROVANT*.wav` | 新歌 |
| `v2_looks.py` LYR / `v2_preview.py` LYR | `research/lyrics_user/ja_en/PROVANT_逐句时间数据.json` | 新歌词包 |
| `lyric_time.py` VOC / `v3_audio.py` SEP | `E:\provant_work\audio\…` | 新干声目录 |
| `hud_v4.py` WHO / `palette.py` CAST + SIG / `hud_v5.py` SIG / `edl_v9.py` SLIT_NAME、NO_COMIC | Fate 角色 | 新作品的角色表、签名色、强调字、不做美漫的角色 |
| `edl_v9.py` SEGS / PB / PA / GLITCH / PUNCH / WALL / HOT_LYR / DIALOG / BAN | PROVANT 段落与钉子 | 新歌的段落简报 |
| `pool_v2.py` HF_ALLOWED / CROP / AB 代码表 | PROVANT 素材 | 新素材 |
| 字体 `assets/fonts` | — | 按 reference/10_assets.md 下载 |

## 最小跑通顺序
`conform_footage → shot_inventory → (子代理打标签) → pool_v2 → adaptive_grid → sep_mdx(歌) → lyric_time → edl_vN → v2_preview KEY analyze → render → encode → v3_audio vN --mux → qc_v1`
| `v5_tape.py` | 警戒线转场层（slit 模式 / cross 模式；角色色胶带 + 核对过的文字） |
| `gen_frame_okita_v2.py` | 冲田卡框 v2（蓝羽织 + 樱花 + 武士刀）生成脚本 |
| `edl_v11.py` | **最新 EDL 生成器**：在 edl_v10 的角色块上加序章（原声咏唱 + 字幕）/ 标题卡 / break（原声整段 + 章节 + 暂停播放器）/ 尾声（歌后台词 + 环境音）；song_map |
| `v6_title.py` | 标题卡（look T）+ 字幕层（serif / burst / chapter / player / **line 台词演出**） |
| `edl_v12.py` | **最新 EDL 生成器**（PROVANT v10）：在 v11 上去掉美漫、break 2 后移 + 回忆对话、源镜头拆分（SPLIT）、台词演出（DLINES）、手标 HUD 框 |
| `sep_bandit.py` | BandIt v2 音乐掩码（维纳软掩码、按需缓存）；`bg()` = 原声背景压低（v4_audio 默认用它，不完全去配乐） |
| `v4_audio.py` | 成片比歌长时的混音：歌分段 + native_audio + 环境底 + 台词 + carve + 限制 |
| `v7_cine.py` | 电影质感收尾层 + 21:9 动态遮幅（break 段） |
| `v8_post.py` | **ER 后处理层**（Edgerunners AMV 移植，通用、分辨率无关、不改 v2_preview 默认行为）：脸保护 / 调色预设（含 d2）/ 色块跳 / 局部光效 / 网点 / 冲击帧 + slam / 反色黑白闪 / 叠画格 + 九宫格拉远 / 对角碎裂 / 叠化 match cut / 画中画色跳 / 自由分格漫画页 / 细线 HUD / 遮幅衔接 + J-cut / 歌词动效预设 / QC；速度线默认关。`python v8_post.py` 自检。见 reference/11_post_er.md |

# Vendored Depth Anything 3

官方源码: https://github.com/ByteDance-Seed/Depth-Anything-3
固定 commit: 44dd68f ("Modify README for installation and model details")

只 vendor `src/depth_anything_3` 包本体 + LICENSE, 供 ai-hub 集群任务
(镜像未预装 depth_anything_3) 通过 `dataloader/modulation_loader.py`
的 `_load_depth_anything3_class()` 直接 import。

运行依赖见仓库根 requirements.txt (omegaconf / addict /
opencv-python-headless / imageio 等)。

更新方式: 同步官方仓库后替换 `src/depth_anything_3/`, 并更新本文件的
commit 记录。

## 本地补丁 (同步上游时需重新应用)

- `api.py`: `utils.export` / `utils.pose_align` 改为函数内延迟 import。
  原顶层 import 会连带加载 pycolmap / moviepy / evo 等导出专用依赖,
  只装了 requirements.txt 推理依赖的集群镜像会在 import api 时失败。
  推理路径 (backfill_cond_depth) 不调用这两处, 行为不变。

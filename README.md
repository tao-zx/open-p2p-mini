# 课题六 · 行动克隆游戏智能体（Open Pixel2Play）

让模型"看"游戏画面、直接输出键鼠动作的视觉-语言-动作（VLA）推理与评测项目。9 天 mini 项目的工程落地，主路径：toy 原始数据 → 预处理 → 200 帧清洗样本 → 推理（150M）→ 预测动作 → 评测 → 指标。

## 目录职责

| 路径 | 职责 |
|---|---|
| `scripts/` | 三个脚本：`preprocess.py`（数据准备）、`infer.py`（推理）、`evaluate.py`（评测） |
| `202500502046-陶子萱-实践报告0X-…/` | 按天的实践报告与阶段文档（需求、选型、约定、报告正文） |
| `项目备忘.md`、`开发日志.md` | 稳定约定与按天日志，下任务前先读 |
| `data/`、`out/`、`pred/`、`checkpoints/` | 数据、中间产物、权重（不提交，见 `.gitignore`） |

## 环境要求

- **preprocess.py（已接真实实现）**：需 **Python 3.11 + OpenCV**（`cv2`，读视频帧）。
- **infer.py / evaluate.py（仍为假实现）**：仅需 Python 3.11 标准库。
- 第 5 天起（infer/evaluate 接真实）改用 **uv** 管理依赖：`uv sync` 安装 PyTorch 与官方依赖。

## 安装依赖

- 当前阶段：跑 preprocess 需装 OpenCV：`pip install opencv-python`（infer/evaluate 假实现无需装）。
- 真实推理阶段（第 5 天）：`uv sync`（届时补 `pyproject.toml` / 依赖清单）。

## 启动 / 首跑

在仓库根目录依次执行三条命令（preprocess 已接真实实现；infer/evaluate 仍为假实现）：

```bash
# 1) 数据准备：真读 toy 数据 → 200 帧清洗样本（帧图片 + 人类动作标注）+ 统计表
python scripts/preprocess.py --data_dir data/toy-examples --out_dir out --n_frames 200

# 2) 推理：生成 200 帧全 0 预测（假实现）
python scripts/infer.py --weights checkpoints/150M/checkpoint-step=00500000.ckpt --samples out/samples.json --out pred

# 3) 评测：输出按键准确率 / 鼠标相关系数（假实现固定 0.0% / 0.0）
python scripts/evaluate.py --pred pred/predictions.json --label out/samples.json
```

期望输出（三步各以 `[OK]` 开头，最后写出 `metrics.json`）：

```
[OK] 处理 200 帧 → 产出 200 条样本
[OK] 写出 out/samples.json、out/stats.csv

[OK] 推理 200 帧完成
[OK] 写出 pred/predictions.json

[OK] 按键准确率 = 0.0% （阈值 55%）
[OK] 鼠标相关系数 = 0.0 （阈值 0.5）
[OK] 写出 metrics.json
```

> 说明：preprocess 已接真实实现（读 `192x192.mp4` + `annotation.proto`，产出 200 帧真样本与真统计表，见 `out/samples.json`、`out/stats.csv`、`out/frames/`）；infer/evaluate 仍为假实现，`--weights` 只是占位参数，`0.0% / 0.0` 也是占位值。第 5 天把 infer/evaluate 换真实实现后，这里才有真实预测与指标。

## 停止

本课题是"运行即结束"的短命令，没有常驻服务；命令跑完进程自然退出，无需额外停止步骤。中途中断按 `Ctrl+C`。

## 环境变量

当前无需任何环境变量。真实推理阶段若官方代码要求（如数据/权重路径），会在本文件补充。

## 更多约定

- 脚本命令、参数、退出码（0=成功 / 2=参数错误 / 3=数据错误）见 `202500502046-陶子萱-实践报告03-数据与约定/接口约定.md`。
- 数据模型与字段见同目录 `数据模型.md`。

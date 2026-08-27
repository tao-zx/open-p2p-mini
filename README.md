# 课题六 · 行为克隆游戏智能体（Open Pixel2Play）

> **说明**：这是**完整项目仓库**的 README。随报告单独提交的 `源码.zip` 是仓库的「代码 + 核心复现文档」子集——其中 `接口约定.md`、`数据模型.md` 平铺在源码包根目录；`项目备忘.md`、`开发日志.md`、各天实践报告，以及 `data/`、`checkpoints/` 等数据产物均不在源码包内（不入库）。

让模型"看"游戏画面、直接输出键鼠动作的视觉-语言-动作（VLA）推理与评测项目。9 天 mini 项目的工程落地，主路径：toy 原始数据 → 预处理 → 200 帧清洗样本 → 推理（150M）→ 预测动作 → 评测 → 指标。

**当前状态（第 9 天 · 结课）**：主路径已端到端贯通，三脚本（preprocess / infer / evaluate）均为真实实现，真实指标 **按键准确率 88.0% / 鼠标相关系数 0.756**（阈值 55% / 0.5，评测 200 帧）；第 8 天回归 smoke 10/10 全过（4 主路径 + 4 异常 + 2 扩展），立项扩展微调四波未达 +8pp/+0.08、已归档为科学发现。第 9 天为结课交付：源码.zip、结课报告（报告09）、演示 PPT + 讲解稿 + 录屏演示。

## 目录职责

| 路径 | 职责 |
|---|---|
| `scripts/` | 六个脚本 + `proto/`（protobuf 定义），各自作用见下一节「脚本总览」 |
| `演示说明.md` | 主路径复现操作文档（环境前提 + 编号步骤 + 预期现象 + 失败检查） |
| `202500502046-陶子萱-实践报告0X-…/` | 按天的实践报告与阶段文档（需求、选型、约定、报告正文） |
| `项目备忘.md`、`开发日志.md` | 稳定约定与按天日志，下任务前先读 |
| `pyproject.toml`、`依赖清单.md` | 依赖固定（`uv sync --extra gpu` 一键复现） |
| `data/`、`out/`、`pred/`、`checkpoints/` | 数据、中间产物、权重（不提交，见 `.gitignore`） |

## 脚本总览（六个脚本怎么串起来）

**主路径**（必做 1–3 的验收链路，`preprocess → infer → evaluate` 顺序执行）：

| 步骤 | 脚本 | 输入 → 输出 | 作用 |
|---|---|---|---|
| ① 数据准备 | `scripts/preprocess.py` | toy 原始数据（`.mp4` + `.proto`）→ `out/samples.json` + `out/stats.csv` | 清洗 200 帧测试集（`system_action` 优先）+ 出统计表 |
| ② 推理 | `scripts/infer.py` | `out/samples.json` + 150M 权重 → `pred/predictions.json` | 模型推理「看画面 → 出键鼠动作」（`teacher_forcing` 验收 / `free_running` 演示） |
| ③ 评测 | `scripts/evaluate.py` | `pred/predictions.json` + `out/samples.json` → `metrics.json` | 算按键准确率（≥55%）/ 鼠标相关系数（≥0.5） |

**扩展路径**（立项扩展：微调 1 款游戏 2 小时）：

| 脚本 | 输入 → 输出 | 作用 |
|---|---|---|
| `scripts/prepare_finetune_data.py` | full-data batch 子集 → `data/codm-2h/`（或 gta-2h） | 切「1 款游戏 2 小时」微调数据，排除测试段防泄漏 |
| `scripts/train.py` | 150M 权重 + 2h 数据 → `out/.../checkpoints/*.ckpt` | 微调（可选冻 image_tokenizer + cosine 调度） |

**通检**（回归兜底）：

| 脚本 | 输入 → 输出 | 作用 |
|---|---|---|
| `scripts/smoke_test.py` | — → 10 条冒烟结果 | 端到端回归：4 主路径 + 4 异常路径 + 2 扩展路径，核对退出码 0/2/3 |

`scripts/proto/` 下 3 个文件（`__init__.py`、`shared_pb2.py`、`video_annotation_pb2.py`）是从官方 open-p2p 复制的 protobuf 定义，供 `preprocess.py` 解析 `annotation.proto`，未做修改。

## 环境要求（真实推理）

真实推理需 **GPU + 官方 open-p2p 依赖**，本机无 N 卡的 Windows 环境无法完整跑通。完整环境准备、数据/权重获取、逐步命令与预期现象见 [`演示说明.md`](演示说明.md)。要点：

- **算力**：GPU 服务器（本课题用 NVIDIA Quadro RTX 6000 24GB）。
- **依赖**：官方 open-p2p 仓库（含 `elefant` 包）+ 其 venv（Python 3.12，torch 2.11+cu128、lightning、torchcodec、protobuf）。本项目依赖固定：`uv sync --extra gpu`（见 `pyproject.toml` + `依赖清单.md`；`elefant` 由 `infer.py --official-repo` 指路）。
- **数据与权重**（运行前提，不入库）：toy 数据 = huggingface 数据集 `elefantai/p2p-toy-examples`（3 段视频 + 标注，约 1GB）；150M 权重 = `elefantai/open-p2p` 的 `checkpoint-step=00500000.ckpt`（2.2GB）+ `model_config.yaml`。下载用 `uv run --with huggingface_hub`（只装 huggingface_hub，绕开官方 pyproject 的 Python 版本锁）；完整下载命令见 `开发日志.md` 第 4 天。
- 三个脚本均只依赖标准库 + 上述官方依赖；`evaluate.py` 仅标准库，可在任意 Python 3.11+ 环境跑。

## 启动 / 首跑

在 GPU 服务器工作目录（如 `/root/workspace/taozixuan/`）依次执行（`python` 用 open-p2p venv）：

```bash
# 1) 数据准备：读 toy 数据 → 200 帧清洗样本（帧图片 + 动作标注）+ 统计表
python scripts/preprocess.py --data_dir data/p2p-toy-examples --out_dir out --n_frames 200

# 2) 推理（teacher_forcing，评测指标用）：加载 150M 权重 → 200 帧预测
python scripts/infer.py --weights checkpoints/150M/checkpoint-step=00500000.ckpt \
  --samples out/samples.json --out pred --official-repo <open-p2p仓库> --mode teacher_forcing

# 3) 评测：算按键准确率 + 鼠标相关系数 → metrics.json
python scripts/evaluate.py --pred pred/predictions.json --label out/samples.json --out metrics.json
```

期望输出（节选，真实结果）：

```
[OK] 处理 200 帧 → 产出 200 条样本
[OK] 推理 200 帧完成
[OK] 按键准确率 = 88.0% （阈值 55%，评测 200 帧）
[OK] 鼠标相关系数 = 0.756 （阈值 0.5，x=0.741 / y=0.772）
```

> `infer.py` 的 `--mode free_running` 用于录屏演示"模型自主运行"（逐帧自主预测、误差累积、key_acc 天然偏低）；验收指标用默认的 `teacher_forcing`。详见 `演示说明.md`。

## 停止与回退

三个脚本均为“运行即结束”的短命令，无常驻服务；跑完进程自然退出，中途中断按 `Ctrl+C`。唯一例外是 `free_running` 模式需约 13 分钟（1600 次前向）。

**回退**：本项目无常驻服务，回退指“换回基线权重”。若微调后权重（`out_gta_*/checkpoints/*.ckpt`）加载报错或指标退化，把 `infer.py` 的 `--weights` 换回 150M 基线权重 `checkpoints/150M/checkpoint-step=00500000.ckpt` 即可恢复基线 88.0% / 0.756。

## 环境变量

当前无需环境变量；数据/权重路径均通过命令行参数传入。

## 更多约定

- 脚本命令、参数、退出码（0=成功 / 2=参数错误 / 3=数据错误）见 `202500502046-陶子萱-实践报告03-数据与约定/接口约定.md`。
- 数据模型与字段见同目录 `数据模型.md`。

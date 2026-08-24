#!/usr/bin/env python3
"""train.py —— 微调（扩展，块 2 真实实现）。

真实实现：加载官方 150M 预训练权重（Stage3LabelledBCLightning.load_from_checkpoint），
在 full-data 的「1 款游戏 2 小时」子集上继续微调（teacher-forcing 行为克隆），产出
微调后权重；再用 infer.py 跑同一测试片段、evaluate.py 对比指标（按键 +8pp 或 鼠标 +0.08）。

关键点（与官方差异，均附出处）：
- 官方 `elefant/policy_model/stage3_finetune.py::train_stage3_finetune` 用
  `_init_stage3_model`（随机权重）初始化，且断言 `stage2_model_path is None`
  （stage3_finetune.py:1057-1064）——即官方这个训练入口**不会**从 150M 权重接着微调
  （跨阶段加载在仓库外的 start_experiment.py 编排脚本里）。本脚本改用官方
  `inference.py:546-550` / `validation.py:119-121` 同款
  `Stage3LabelledBCLightning.load_from_checkpoint` 加载 150M 权重，再 trainer.fit 续训。
- 默认关闭 torch.compile（`set_stance("force_eager")`）：服务器 venv 缺 Python.h，
  flex_attention 的 CUDA 扩展 JIT 会崩（与 infer.py 的 `_stage3.compile=False` 同理，
  对齐官方 train.py 的 `--no_compile`，见 elefant/policy_model/train.py:31-33）。
- 单 GPU、bf16-mixed、关 wandb（本地训练日志用 CSVLogger；官方多卡 DDP 会与
  accumulate_grad_batches 死锁，见 stage3_finetune.py:1083-1085 注释）。

约定见 接口约定.md；退出码 0=成功 / 2=参数错误 / 3=数据错误。
"""
import argparse
import json
import os
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser(description="150M 权重 + 2h 数据子集 → 微调权重")
    parser.add_argument("--weights", required=True, help="150M 预训练权重（.ckpt，微调起点）")
    parser.add_argument("--data_dir", required=True,
                        help="微调数据目录（1 款游戏 2h 子集，含 .mp4 + .proto）")
    parser.add_argument("--out_dir", required=True, help="输出目录（微调权重 + 训练日志）")
    parser.add_argument("--official-repo", default=None,
                        help="官方 open-p2p 仓库根目录（含 elefant 包）；已 pip install -e 则可省略")
    parser.add_argument("--config", default=None,
                        help="配置 yaml；默认取官方 config/policy_model/150M.yaml")
    parser.add_argument("--n_steps", type=int, default=1000, help="微调步数（默认 1000）")
    parser.add_argument("--freeze_steps", type=int, default=0,
                        help="冻结 transformer 层的前 N 步（0=全量微调，>0 先只训动作头再解冻）")
    parser.add_argument("--save_every", type=int, default=100, help="每多少步存一次 checkpoint")
    parser.add_argument("--compile", action="store_true",
                        help="启用 torch.compile（默认关闭走 eager，避免服务器缺 Python.h 的 JIT 崩）")
    return parser.parse_args(argv)


def validate_args(args):
    if not os.path.isfile(args.weights):
        print(f"[ERROR] 权重文件不存在: {args.weights}（退出码 3）", file=sys.stderr)
        return 3
    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] 数据目录不存在: {args.data_dir}（退出码 3）", file=sys.stderr)
        return 3
    if args.official_repo and not os.path.isdir(args.official_repo):
        print(f"[ERROR] 官方仓库不存在: {args.official_repo}（退出码 3）", file=sys.stderr)
        return 3
    if args.n_steps <= 0:
        print(f"[ERROR] --n_steps 必须是正整数，收到 {args.n_steps}（退出码 2）", file=sys.stderr)
        return 2
    if args.freeze_steps < 0:
        print(f"[ERROR] --freeze_steps 不能为负，收到 {args.freeze_steps}（退出码 2）", file=sys.stderr)
        return 2
    return 0


def find_protos(data_dir):
    """递归扫 data_dir 下所有 .proto（与官方 video_proto_dataset.py:398-405 同款
    os.walk 递归），返回排序后的绝对路径，保证可复现。"""
    protos = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".proto"):
                protos.append(os.path.join(root, f))
    return sorted(protos)


def resolve_config(args):
    if args.config:
        return args.config
    if args.official_repo:
        guess = os.path.join(args.official_repo, "config", "policy_model", "150M.yaml")
        if os.path.isfile(guess):
            return guess
    print("[ERROR] 未找到配置 yaml：--config 未给，且官方 150M.yaml 不存在（退出码 3）", file=sys.stderr)
    return None


def run(args):
    err = validate_args(args)
    if err:
        return err

    protos = find_protos(args.data_dir)
    if not protos:
        print(f"[ERROR] 未在 {args.data_dir} 找到 .proto（退出码 3）", file=sys.stderr)
        return 3
    print(f"[OK] 数据目录含 {len(protos)} 个 .proto（{args.data_dir}）")

    config_yaml = resolve_config(args)
    if config_yaml is None:
        return 3

    # 延迟 import elefant（真实训练依赖官方仓库 + GPU）
    if args.official_repo:
        sys.path.insert(0, os.path.abspath(args.official_repo))
    try:
        import torch
        import lightning as pl
        from elefant.config import load_config
        from elefant.policy_model.config import LightningPolicyConfig
        from elefant.policy_model.stage3_finetune import (
            Stage3DataModule,
            Stage3LabelledBCLightning,
        )
    except ImportError as e:
        print(f"[ERROR] 缺少官方 elefant 依赖: {e}（退出码 3）", file=sys.stderr)
        print("[ERROR] 微调需在装有官方 open-p2p 仓库的 Linux/GPU 环境运行，可用 --official-repo 指定。",
              file=sys.stderr)
        return 3

    if not torch.cuda.is_available():
        print("[ERROR] 未检测到 CUDA GPU，微调需 GPU（退出码 3）", file=sys.stderr)
        return 3

    # 关闭 torch.compile（必须在模型 __init__ 之前设，模型构造里的 torch.compile 才会走 no-op）
    if not args.compile:
        torch.compiler.set_stance("force_eager")
        print("[OK] torch.compile 已关闭（force_eager）")

    config = load_config(config_yaml, LightningPolicyConfig)

    # 覆盖：数据路径 / 训练步数 / 冻结步数 / 输出路径 / 关 wandb
    config.stage3_finetune.training_dataset.local_prefix = args.data_dir
    for vd in config.stage3_finetune.validation_datasets:
        vd.local_prefix = args.data_dir
    config.stage3_finetune.n_training_steps = args.n_steps
    config.stage3_finetune.freeze_transformer_layers_for_steps = args.freeze_steps
    config.shared.output_path = args.out_dir
    config.wandb.enabled = False

    datamodule = Stage3DataModule(config)

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, "checkpoints"),
        every_n_train_steps=args.save_every,
        filename="checkpoint-{step:08d}",
        save_top_k=-1,
    )
    logger = pl.pytorch.loggers.CSVLogger(save_dir=args.out_dir, name="logs")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=1,
        max_steps=args.n_steps,
        logger=logger,
        precision=config.shared.precision,
        num_sanity_val_steps=0,
    )

    # 从 150M checkpoint 加载（微调起点），用 trainer.init_module 放到 GPU + 正确精度，
    # 对齐官方 inference.py:546-550 与 train_stage3_finetune.py:1299-1300。
    with trainer.init_module():
        model = Stage3LabelledBCLightning.load_from_checkpoint(args.weights, config=config)
    print(f"[OK] 从 150M 权重加载模型: {args.weights}")

    print(f"[OK] 开始微调：{args.n_steps} 步，数据 {args.data_dir}")
    trainer.fit(model, datamodule)

    # 保存最终权重 + 摘要（finetuned.ckpt 用同一 config 可被 infer.py load_from_checkpoint）
    final_path = os.path.join(args.out_dir, "finetuned.ckpt")
    trainer.save_checkpoint(final_path)
    summary = {
        "base_weights": args.weights,
        "data_dir": args.data_dir,
        "n_protos": len(protos),
        "n_steps": args.n_steps,
        "freeze_steps": args.freeze_steps,
        "config": config_yaml,
        "final_checkpoint": final_path,
    }
    summary_path = os.path.join(args.out_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] 微调完成，最终权重 {final_path}")
    print(f"[OK] 写出 {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))
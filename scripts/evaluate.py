#!/usr/bin/env python3
"""evaluate.py —— 评测（主路径第三步）。

# TODO(假实现)：不真正比对预测与标注，固定输出 0.0% / 0.0。
第 5 天替换为真实实现：按键准确率、鼠标相关系数（位移相关）。
约定见 接口约定.md；退出码 0=成功 / 2=参数错误 / 3=数据错误。
"""
import argparse
import json
import os
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser(description="预测 + 标注 → 按键准确率 / 鼠标相关系数")
    parser.add_argument("--pred", required=True, help="infer 产出的 predictions.json")
    parser.add_argument("--label", required=True, help="preprocess 产出的 samples.json（含真人标注）")
    return parser.parse_args(argv)


def run(args):
    # TODO(假实现)：只做文件/帧数校验 + 固定指标，不做真实比对。
    if not os.path.isfile(args.pred):
        print(f"[ERROR] 预测文件不存在: {args.pred}（退出码 3）", file=sys.stderr)
        return 3
    if not os.path.isfile(args.label):
        print(f"[ERROR] 标注文件不存在: {args.label}（退出码 3）", file=sys.stderr)
        return 3

    with open(args.pred, "r", encoding="utf-8") as f:
        pred = json.load(f)
    with open(args.label, "r", encoding="utf-8") as f:
        label = json.load(f)

    n_pred = pred.get("n_frames", 0)
    n_label = label.get("n_frames", 0)
    if n_pred != n_label:
        print(f"[ERROR] 预测 {n_pred} 帧 vs 标注 {n_label} 帧，帧数不一致（退出码 3）", file=sys.stderr)
        return 3

    # TODO(假实现)：固定指标 0.0% / 0.0。
    metrics = {
        "key_accuracy": 0.0,
        "mouse_correlation": 0.0,
        "n_samples": n_label,
        "action_distribution": {},
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[OK] 按键准确率 = 0.0% （阈值 55%）")
    print("[OK] 鼠标相关系数 = 0.0 （阈值 0.5）")
    print("[OK] 写出 metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))

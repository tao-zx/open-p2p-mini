#!/usr/bin/env python3
"""infer.py —— 推理（主路径最关键一步）。

# TODO(假实现)：不加载 150M 权重，固定返回全 0 动作。
第 5 天替换为真实实现：加载 150M 模型，对每帧输出 keys/mouse_buttons/mouse_delta_x/mouse_delta_y。
约定见 接口约定.md；退出码 0=成功 / 2=参数错误 / 3=数据错误。
"""
import argparse
import json
import os
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser(description="150M 权重 + 样本 → 预测动作")
    parser.add_argument("--weights", required=True, help="150M 模型权重文件路径")
    parser.add_argument("--samples", required=True, help="preprocess 产出的 samples.json")
    parser.add_argument("--out", required=True, help="预测输出目录")
    return parser.parse_args(argv)


def run(args):
    # TODO(假实现)：不加载 --weights，只读 --samples 的帧数，造全 0 预测。
    if not os.path.isfile(args.samples):
        print(f"[ERROR] 样本文件不存在: {args.samples}（退出码 3）", file=sys.stderr)
        return 3

    with open(args.samples, "r", encoding="utf-8") as f:
        samples = json.load(f)
    n_frames = samples.get("n_frames", 0)

    # TODO(假实现)：全 0 动作。
    frames = [
        {"frame_id": i, "keys": [], "mouse_buttons": [], "mouse_delta_x": 0, "mouse_delta_y": 0}
        for i in range(n_frames)
    ]
    pred = {"n_frames": n_frames, "frames": frames}

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False, indent=2)

    print(f"[OK] 推理 {n_frames} 帧完成")
    print(f"[OK] 写出 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))

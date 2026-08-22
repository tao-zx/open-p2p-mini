#!/usr/bin/env python3
"""preprocess.py —— 数据准备（主路径第一步）。

# TODO(假实现)：不真正读取 toy 数据，固定返回 200 条空样本（动作全 0）。
第 5 天替换为真实实现：从 --data_dir 读 .mp4 + .proto，切 200 帧 + 逐帧人类动作标注。
约定见 接口约定.md；退出码 0=成功 / 2=参数错误 / 3=数据错误。
"""
import argparse
import csv
import json
import os
import sys


def parse_args(argv):
    # 参数签名与真实实现一致（--data_dir / --out_dir / --n_frames）。
    parser = argparse.ArgumentParser(description="原始数据 → 200 帧清洗样本 + 统计表")
    parser.add_argument("--data_dir", required=True, help="toy 数据目录（含 .mp4 + .proto）")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--n_frames", type=int, default=200, help="测试集帧数，默认 200")
    return parser.parse_args(argv)


def run(args):
    # TODO(假实现)：只做参数校验 + 造 200 条空样本，不读取 --data_dir。
    if args.n_frames <= 0:
        print(f"[ERROR] --n_frames 必须是正整数，收到 {args.n_frames}（退出码 2）", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    # TODO(假实现)：空样本，动作全 0。
    frames = [
        {"frame_id": i, "keys": [], "mouse_buttons": [], "mouse_delta_x": 0, "mouse_delta_y": 0}
        for i in range(args.n_frames)
    ]
    samples = {"n_frames": args.n_frames, "frames": frames}

    samples_path = os.path.join(args.out_dir, "samples.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    stats_path = os.path.join(args.out_dir, "stats.csv")
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_frames", "key_presses", "mouse_clicks"])
        writer.writerow([args.n_frames, 0, 0])

    print(f"[OK] 处理 {args.n_frames} 帧 → 产出 {args.n_frames} 条样本")
    print(f"[OK] 写出 {samples_path}、{stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))

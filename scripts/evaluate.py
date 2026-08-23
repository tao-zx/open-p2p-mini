#!/usr/bin/env python3
"""evaluate.py —— 评测（主路径第三步，真实实现）。

真实实现：比对 predictions.json（模型预测）与 samples.json（真人标注），
算两个验收指标：
- 按键准确率 key_accuracy：逐帧 keys 集合精确匹配的占比（阈值 ≥55%）；
- 鼠标相关系数 mouse_correlation：预测 vs 真人 mouse_delta 的 Pearson 相关系数，
  取 x、y 两轴相关系数的平均（阈值 ≥0.5）。

口径（见 数据模型.md 说明 2）：
- 只取 user_action（人类动作），is_known=false 的帧未可靠标注，评测时跳过；
- 空帧（无按键无鼠标位移）照常参与按键准确率（预测"空"=正确），也参与鼠标相关。

约定见 接口约定.md；退出码 0=成功 / 2=参数错误 / 3=数据错误。
"""
import argparse
import json
import os
import sys
from collections import Counter


def parse_args(argv):
    parser = argparse.ArgumentParser(description="预测 + 标注 → 按键准确率 / 鼠标相关系数")
    parser.add_argument("--pred", required=True, help="infer 产出的 predictions.json")
    parser.add_argument("--label", required=True, help="preprocess 产出的 samples.json（含真人标注）")
    parser.add_argument("--out", default="metrics.json", help="指标输出路径，默认 metrics.json")
    return parser.parse_args(argv)


def pearson(xs, ys):
    """Pearson 相关系数；任一侧无方差（如全 0）时无定义，返回 0.0。"""
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def keys_exact_match(pred_keys, label_keys):
    """按键准确率口径一：该帧 keys 集合与真人完全一致（顺序无关）。"""
    return set(pred_keys) == set(label_keys)


def run(args):
    for path, name in ((args.pred, "预测"), (args.label, "标注")):
        if not os.path.isfile(path):
            print(f"[ERROR] {name}文件不存在: {path}（退出码 3）", file=sys.stderr)
            return 3

    with open(args.pred, "r", encoding="utf-8") as f:
        pred = json.load(f)
    with open(args.label, "r", encoding="utf-8") as f:
        label = json.load(f)

    pred_frames = pred.get("frames", [])
    label_frames = label.get("frames", [])

    # 以 frame_id 对齐；只保留 is_known != false 的标注帧（不可靠标注跳过）
    label_by_id = {
        fr["frame_id"]: fr
        for fr in label_frames
        if fr.get("is_known", True) is not False
    }

    n_correct = 0
    n_scored = 0
    dx_pred, dx_label = [], []
    dy_pred, dy_label = [], []
    key_counter = Counter()
    mouse_counter = Counter()

    for pf in pred_frames:
        fid = pf.get("frame_id")
        lf = label_by_id.get(fid)
        if lf is None:
            continue  # 该帧无可靠标注，跳过
        n_scored += 1
        if keys_exact_match(pf.get("keys", []), lf.get("keys", [])):
            n_correct += 1
        dx_pred.append(pf.get("mouse_delta_x", 0))
        dx_label.append(lf.get("mouse_delta_x", 0))
        dy_pred.append(pf.get("mouse_delta_y", 0))
        dy_label.append(lf.get("mouse_delta_y", 0))
        key_counter.update(lf.get("keys", []))
        mouse_counter.update(lf.get("mouse_buttons", []))

    if n_scored == 0:
        print("[ERROR] 没有可评测的帧（标注帧全部 is_known=false？）（退出码 3）", file=sys.stderr)
        return 3

    key_accuracy = n_correct / n_scored
    mouse_x_corr = pearson(dx_pred, dx_label)
    mouse_y_corr = pearson(dy_pred, dy_label)
    mouse_correlation = (mouse_x_corr + mouse_y_corr) / 2.0

    metrics = {
        "key_accuracy": round(key_accuracy, 4),
        "mouse_correlation": round(mouse_correlation, 4),
        "mouse_x_corr": round(mouse_x_corr, 4),
        "mouse_y_corr": round(mouse_y_corr, 4),
        "n_samples": n_scored,
        "action_distribution": {
            "keys": {k: v for k, v in sorted(key_counter.items())},
            "mouse_buttons": {k: v for k, v in sorted(mouse_counter.items())},
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] 按键准确率 = {key_accuracy * 100:.1f}% （阈值 55%，评测 {n_scored} 帧）")
    print(f"[OK] 鼠标相关系数 = {mouse_correlation:.3f} （阈值 0.5，x={mouse_x_corr:.3f} / y={mouse_y_corr:.3f}）")
    print(f"[OK] 写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))

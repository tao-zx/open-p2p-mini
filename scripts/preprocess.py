#!/usr/bin/env python3
"""preprocess.py —— 数据准备（主路径第一步）。

真实实现：从 --data_dir 读每段视频的 192x192.mp4 + annotation.proto，
按各视频帧数比例均匀采样 --n_frames 帧，切出帧图片 + 逐帧人类动作标注（user_action）。
约定见 接口约定.md；退出码 0=成功 / 2=参数错误 / 3=数据错误。
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import cv2

# 引入官方 proto schema（复制进 scripts/proto/，出处见该目录 __init__.py）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto import video_annotation_pb2  # noqa: E402

FRAME_SIZE = 192  # 帧尺寸固定 192×192，写死（接口约定：不作参数）


def parse_args(argv):
    parser = argparse.ArgumentParser(description="原始数据 → 200 帧清洗样本 + 统计表")
    parser.add_argument("--data_dir", required=True, help="toy 数据目录（含 .mp4 + .proto）")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--n_frames", type=int, default=200, help="测试集帧数，默认 200")
    return parser.parse_args(argv)


def find_videos(data_dir):
    """扫描 data_dir，返回按路径排序的 [(video_path, annotation_path)]，保证可复现。"""
    pairs = []
    for root, _dirs, files in os.walk(data_dir):
        if "192x192.mp4" in files and "annotation.proto" in files:
            pairs.append(
                (os.path.join(root, "192x192.mp4"), os.path.join(root, "annotation.proto"))
            )
    return sorted(pairs)


def parse_annotation(proto_path):
    """解析 annotation.proto，返回 (metadata, frame_annotations 列表)。"""
    va = video_annotation_pb2.VideoAnnotation()
    with open(proto_path, "rb") as f:
        va.ParseFromString(f.read())
    return va.metadata, list(va.frame_annotations)


def even_indices(total, k):
    """在 [0, total-1] 内均匀取 k 个整数下标（含首尾，四舍五入去重）。"""
    k = min(k, total)
    if k <= 0 or total <= 0:
        return []
    if k == 1:
        return [total // 2]
    step = (total - 1) / (k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


def distribute(n_frames, frame_counts):
    """按各视频帧数比例分配 n_frames（最大余数法，保证总和恰为 n_frames）。"""
    total = sum(frame_counts)
    if total <= 0:
        return [0] * len(frame_counts)
    exact = [n_frames * c / total for c in frame_counts]
    counts = [int(x) for x in exact]
    order = sorted(range(len(frame_counts)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in range(n_frames - sum(counts)):
        counts[order[i]] += 1
    return counts


def extract_action(frame_annotation):
    """从一帧 FrameAnnotation 取人类动作（user_action）。本课题忽略 system_action。"""
    ua = frame_annotation.user_action
    return {
        "keys": list(ua.keyboard.keys),
        "mouse_buttons": list(ua.mouse.buttons_down),
        "mouse_delta_x": ua.mouse.mouse_delta_px.x,
        "mouse_delta_y": ua.mouse.mouse_delta_px.y,
        "is_user_action": True,  # 只取 user_action（人类动作），见 数据模型.md 说明 2
        "is_known": ua.is_known,  # false=该帧动作未可靠标注，评测时跳过
    }


def run(args):
    if args.n_frames <= 0:
        print(f"[ERROR] --n_frames 必须是正整数，收到 {args.n_frames}（退出码 2）", file=sys.stderr)
        return 2
    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] 数据目录不存在: {args.data_dir}（退出码 3）", file=sys.stderr)
        return 3

    videos = find_videos(args.data_dir)
    if not videos:
        print(f"[ERROR] 未在 {args.data_dir} 找到 192x192.mp4 + annotation.proto（退出码 3）", file=sys.stderr)
        return 3

    # 1) 先读每段视频的帧数与标注，用于按比例分配
    per_video = []  # [(video_path, annotation_path, frame_annotations, total_frames)]
    for video_path, proto_path in videos:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 打不开视频: {video_path}（退出码 3）", file=sys.stderr)
            return 3
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        _meta, anns = parse_annotation(proto_path)
        if total_frames <= 0:
            total_frames = len(anns)  # 兜底：标注数即帧数
        per_video.append((video_path, proto_path, anns, total_frames))

    alloc = distribute(args.n_frames, [pv[3] for pv in per_video])

    # 2) 逐视频顺序读帧（顺序读保证 frame_index 与画面精确对应），切帧 + 取标注
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    frames = []
    frame_id = 0
    for (video_path, proto_path, anns, total_frames), k in zip(per_video, alloc):
        if k <= 0:
            continue
        want = even_indices(total_frames, k)
        if not want:
            continue
        max_idx = want[-1]
        rel_video = os.path.relpath(video_path, args.data_dir).replace(os.sep, "/")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 打不开视频: {video_path}（退出码 3）", file=sys.stderr)
            return 3
        i = 0
        while i <= max_idx:
            ok, img = cap.read()
            if not ok or img is None:
                break
            if i in want:
                h, w = img.shape[:2]
                if (h, w) != (FRAME_SIZE, FRAME_SIZE):
                    img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE))
                frame_name = f"{frame_id:04d}.png"
                frame_path = f"frames/{frame_name}"  # 存 JSON 用正斜杠，跨平台可复现
                cv2.imwrite(os.path.join(args.out_dir, "frames", frame_name), img)
                ann = anns[i] if i < len(anns) else None
                if ann is None:
                    print(f"[WARN] {rel_video} 第 {i} 帧无标注，跳过", file=sys.stderr)
                else:
                    frames.append(
                        {
                            "frame_id": frame_id,
                            "video": rel_video,
                            "frame_index": i,
                            "frame_path": frame_path,  # 相对 --out_dir
                            **extract_action(ann),
                        }
                    )
                    frame_id += 1
            i += 1
        cap.release()

    if not frames:
        print("[ERROR] 未能切出任何帧（退出码 3）", file=sys.stderr)
        return 3

    # 3) 写 samples.json
    samples = {"n_frames": len(frames), "frames": frames}
    samples_path = os.path.join(args.out_dir, "samples.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 4) 写 stats.csv（样本数 + 动作分布）
    key_counter = Counter()
    mouse_counter = Counter()
    n_key_frames = n_mouse_click_frames = n_mouse_move_frames = n_empty = 0
    for fr in frames:
        if fr["keys"]:
            n_key_frames += 1
        if fr["mouse_buttons"]:
            n_mouse_click_frames += 1
        if fr["mouse_delta_x"] != 0 or fr["mouse_delta_y"] != 0:
            n_mouse_move_frames += 1
        if not fr["keys"] and not fr["mouse_buttons"] and fr["mouse_delta_x"] == 0 and fr["mouse_delta_y"] == 0:
            n_empty += 1
        key_counter.update(fr["keys"])
        mouse_counter.update(fr["mouse_buttons"])

    stats_path = os.path.join(args.out_dir, "stats.csv")
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_frames", "n_key_frames", "n_mouse_click_frames", "n_mouse_move_frames", "n_empty_frames"])
        w.writerow([len(frames), n_key_frames, n_mouse_click_frames, n_mouse_move_frames, n_empty])
        w.writerow([])
        w.writerow(["key", "count"])
        for key, cnt in sorted(key_counter.items()):
            w.writerow([key, cnt])
        w.writerow([])
        w.writerow(["mouse_button", "count"])
        for btn, cnt in sorted(mouse_counter.items()):
            w.writerow([btn, cnt])

    print(f"[OK] 处理 {len(frames)} 帧 → 产出 {len(frames)} 条样本")
    print(f"[OK] 写出 {samples_path}、{stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))

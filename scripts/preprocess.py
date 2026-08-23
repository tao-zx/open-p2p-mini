#!/usr/bin/env python3
"""preprocess.py —— 数据准备（主路径第一步）。

真实实现：从 --data_dir 选 1 个 run（单一游戏），读 192x192.mp4 + annotation.proto，
连续取前 --n_frames 帧作为测试片段，切出帧图片 + 逐帧动作标注（system_action 优先，对齐官方）。
约定见 接口约定.md；退出码 0=成功 / 2=参数错误 / 3=数据错误。
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

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


def extract_action(frame_annotation):
    """从一帧 FrameAnnotation 取动作标注，system_action 优先（对齐官方
    action_label_video_proto_dataset.py：system 已知用 system，否则回退 user，
    都未知则空动作）。"""
    ua = frame_annotation.user_action
    sa = frame_annotation.system_action
    if sa.is_known:
        src, is_user = sa, False
    elif ua.is_known:
        src, is_user = ua, True
    else:
        src, is_user = None, False
    return {
        "keys": list(src.keyboard.keys) if src else [],
        "mouse_buttons": list(src.mouse.buttons_down) if src else [],
        "mouse_delta_x": src.mouse.mouse_delta_px.x if src else 0,
        "mouse_delta_y": src.mouse.mouse_delta_px.y if src else 0,
        "is_user_action": is_user,
        "is_known": sa.is_known or ua.is_known,  # false=该帧动作未可靠标注，评测时跳过
    }


def read_video_frames(video_path, n_take):
    """用 torchcodec（官方读帧工具，输出 RGB uint8）读 mp4 前 n_take 帧。

    返回 (frames_np, total_frames)：frames_np 形状 (n, H, W, 3) uint8 RGB。
    torchcodec 只在真实读帧时才 import（本机可能未装，参数校验不依赖它）。
    """
    import numpy as np
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path), device="cpu", num_ffmpeg_threads=1)
    total = len(decoder)
    n = min(n_take, total)
    frames_t = decoder[0:n]  # (n, C, H, W) uint8 RGB
    frames_t = frames_t.permute(0, 2, 3, 1)  # (n, H, W, C)
    return frames_t.numpy(), total


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

    # 选 1 个 run（sorted 保证可复现）作为"单一游戏测试集"，连续取前 n_frames 帧。
    # 对齐课题"测试集 200 帧（与训练/微调数据隔离，Toy 或所选单一游戏）"与官方口径，
    # 而非多视频均匀采样（均匀采样破坏帧间时序，会拉低行为克隆评估指标）。
    video_path, proto_path = videos[0]
    _meta, anns = parse_annotation(proto_path)
    try:
        frames_np, total_frames = read_video_frames(video_path, args.n_frames)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] 读视频失败: {video_path}（{e}，退出码 3）", file=sys.stderr)
        return 3
    if total_frames <= 0:
        total_frames = len(anns)  # 兜底：标注数即帧数
    n_take = min(args.n_frames, total_frames, len(anns), len(frames_np))
    rel_video = os.path.relpath(video_path, args.data_dir).replace(os.sep, "/")

    # 顺序读帧（torchcodec 输出 RGB uint8，顺序保证 frame_index 与画面精确对应），切帧 + 取标注
    import numpy as np
    from PIL import Image

    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    frames = []
    frame_id = 0
    for i in range(n_take):
        img = frames_np[i]  # (H, W, 3) RGB uint8
        if img.shape[0] != FRAME_SIZE or img.shape[1] != FRAME_SIZE:
            img = np.asarray(Image.fromarray(img).resize((FRAME_SIZE, FRAME_SIZE)))
        frame_name = f"{frame_id:04d}.png"
        frame_path = f"frames/{frame_name}"  # 存 JSON 用正斜杠，跨平台可复现
        Image.fromarray(img).save(os.path.join(args.out_dir, "frames", frame_name))
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

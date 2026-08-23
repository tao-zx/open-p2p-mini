#!/usr/bin/env python3
"""infer.py —— 推理（主路径最关键一步，真实实现）。

真实实现：加载官方 150M 权重（Stage3LabelledBCLightning），按模式输出预测动作：
- teacher_forcing（默认，评测指标用）：把 GT 真实动作填进历史窗口，一次前向预测
  全部帧再解码。这是"行为克隆标准评估"，产出供 evaluate 算按键准确率/鼠标相关系数
  的 predictions.json（课题验收的 55%/0.5 阈值即此口径）。
- free_running（流式演示用）：维护 200 帧滚动上下文 + 自回归动作解码，逐帧自主
  预测（误差会累积，key_acc 天然低于 teacher_forcing，仅用于第 5 天录屏演示）。

说明：不 import 官方 elefant.policy_model.inference —— 该模块在 Python 3.12 下
有 AsyncGenerator 注解兼容 bug（会整模块求值崩溃）；这里直接复用官方
Stage3LabelledBCLightning.online_full_predict 重建等价的喂帧循环，逻辑对照官方
FullInferenceState（自回归逐 token 填动作）。

运行前提（与假实现不同，真实推理必须在 GPU / Linux 环境）：
- 装有官方 open-p2p 仓库（需 import `elefant`），用 --official-repo 指向仓库根目录；
- torch + CUDA GPU；150M 权重为 CUDA 权重，CPU 极慢。

约定见 接口约定.md；退出码 0=成功 / 2=参数错误 / 3=数据错误。
"""
import argparse
import json
import os
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser(description="150M 权重 + 样本 → 预测动作（真实推理）")
    parser.add_argument("--weights", required=True, help="150M 模型权重文件路径（.ckpt）")
    parser.add_argument("--samples", required=True, help="preprocess 产出的 samples.json")
    parser.add_argument("--out", required=True, help="预测输出目录")
    parser.add_argument("--config", default=None,
                        help="150M 配置 yaml；默认取 --weights 同目录下的 model_config.yaml")
    parser.add_argument("--official-repo", default=None,
                        help="官方 open-p2p 仓库根目录（含 elefant 包）；已 pip install -e 则可省略")
    parser.add_argument("--mode", default="teacher_forcing",
                        choices=["teacher_forcing", "free_running"],
                        help="teacher_forcing=评测指标用（默认，GT 动作填历史一次前向）；"
                             "free_running=流式演示用（逐帧自主预测，误差累积）")
    return parser.parse_args(argv)


def resolve_config(args):
    if args.config:
        return args.config
    guess = os.path.join(os.path.dirname(os.path.abspath(args.weights)), "model_config.yaml")
    if os.path.isfile(guess):
        return guess
    print(f"[ERROR] 未找到配置 yaml：--config 未给，且 {guess} 不存在（退出码 3）", file=sys.stderr)
    return None


def load_samples(args):
    if not os.path.isfile(args.weights):
        print(f"[ERROR] 权重文件不存在: {args.weights}（退出码 3）", file=sys.stderr)
        return None
    if not os.path.isfile(args.samples):
        print(f"[ERROR] 样本文件不存在: {args.samples}（退出码 3）", file=sys.stderr)
        return None
    with open(args.samples, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frames_as_tensor(samples, base_dir):
    """读 samples.json 里的 frame_path（相对 base_dir），返回 (T,3,192,192) uint8 numpy。"""
    import numpy as np
    from PIL import Image

    frames = []
    for fr in samples["frames"]:
        p = os.path.join(base_dir, fr["frame_path"])
        if not os.path.isfile(p):
            print(f"[ERROR] 帧图片不存在: {p}（退出码 3）", file=sys.stderr)
            return None
        img = Image.open(p).convert("RGB")
        if img.size != (192, 192):
            img = img.resize((192, 192))
        arr = np.asarray(img, dtype=np.uint8)  # (H,W,3)
        arr = arr.transpose(2, 0, 1)  # (3,H,W)
        frames.append(arr)
    return np.stack(frames, axis=0)  # (T,3,192,192)


def _predict_teacher_forcing(model, config, action_mapping, frames_np, samples):
    """teacher-forcing 推理（行为克隆标准评估）：把 GT 真实动作填进历史窗口，
    一次前向预测全部帧，再解码。与官方 action_label_video_proto_dataset.py 的
    动作口径一致（preprocess 已按 system_action 优先产出 samples.json）。"""
    import torch
    from torch.utils import _pytree as pytree
    from elefant.data.proto import shared_pb2

    T = config.shared.n_seq_timesteps
    device = model.device
    mouse_approach = config.inference.mouse_sampling_approach
    N = frames_np.shape[0]

    frame_in = torch.zeros((1, T, 3, 192, 192), dtype=torch.uint8, device=device)
    frame_in[:, :N] = torch.from_numpy(frames_np).to(device).unsqueeze(0)

    action_in = action_mapping.make_empty_action(T)
    action_in = pytree.tree_map(lambda x: x.unsqueeze(0).to(device), action_in)
    for i, fr in enumerate(samples["frames"]):
        v = shared_pb2.Vec2Int(x=int(fr["mouse_delta_x"]), y=int(fr["mouse_delta_y"]))
        gt = action_mapping.action_to_tensor(
            list(fr["keys"]), list(fr["mouse_buttons"]), v
        )
        action_in.keys[:, i] = gt.keys
        action_in.mouse_buttons[:, i] = gt.mouse_buttons
        action_in.mouse_delta_x[:, i] = gt.mouse_delta_x
        action_in.mouse_delta_y[:, i] = gt.mouse_delta_y

    text_embed_dim = model._get_text_embedding_dim()
    text_embed = torch.zeros((1, T, 1, text_embed_dim), dtype=torch.bfloat16, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        action_out, _ = model.online_full_predict(
            frames=frame_in, actions=action_in,
            sampling_temperature=config.inference.sampling_temperature,
            text_tokens_embed=text_embed,
        )

    preds = []
    for i, fr in enumerate(samples["frames"]):
        step = type(action_out)(
            keys=action_out.keys[:, i, :],
            mouse_buttons=action_out.mouse_buttons[:, i, :],
            mouse_delta_x=action_out.mouse_delta_x[:, i, :],
            mouse_delta_y=action_out.mouse_delta_y[:, i, :],
        )
        act = action_mapping.tensor_to_action(step, mouse_sampling_approach=mouse_approach)
        preds.append({
            "frame_id": fr["frame_id"],
            "keys": list(act.keys),
            "mouse_buttons": list(act.mouse_buttons),
            "mouse_delta_x": int(act.mouse_delta_x),
            "mouse_delta_y": int(act.mouse_delta_y),
        })
    return preds


def _write_predictions(args, samples, preds, n_frames):
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_frames": n_frames, "frames": preds}, f, ensure_ascii=False, indent=2)
    print(f"[OK] 推理 {n_frames} 帧完成")
    print(f"[OK] 写出 {out_path}")


def run(args):
    samples = load_samples(args)
    if samples is None:
        return 3
    config_yaml = resolve_config(args)
    if config_yaml is None:
        return 3

    base_dir = os.path.dirname(os.path.abspath(args.samples))
    frames_np = load_frames_as_tensor(samples, base_dir)
    if frames_np is None:
        return 3
    n_frames = frames_np.shape[0]

    # 延迟 import elefant（真实推理依赖官方仓库 + GPU）
    if args.official_repo:
        sys.path.insert(0, os.path.abspath(args.official_repo))
    try:
        import torch
        import lightning as pl
        from torch.utils import _pytree as pytree
        from elefant.config import load_config
        from elefant.policy_model.config import LightningPolicyConfig
        import elefant.policy_model.stage3_finetune as _stage3
        from elefant.policy_model.stage3_finetune import Stage3LabelledBCLightning
        from elefant.data.action_mapping import UniversalAutoregressiveActionMapping
    except ImportError as e:
        print(f"[ERROR] 缺少官方 elefant 依赖: {e}（退出码 3）", file=sys.stderr)
        print("[ERROR] 真实推理需在装有官方 open-p2p 仓库的 Linux/GPU 环境运行，可用 --official-repo 指定。", file=sys.stderr)
        return 3

    # 关闭 torch.compile：官方 online_full_predict 里 `compile` 是裸名字（无同名参数/模块级变量，
    # 实际解析到 Python 内建 compile()，恒真 → 永远走 torch.compile(fullgraph=True)）。
    # 服务器 venv 缺 Python.h，flex_attention 的 CUDA 扩展 JIT 编译会崩；注入模块级 compile=False 走 eager
    # （与官方离线推理在不编译时一致的路径）。
    _stage3.compile = False

    config = load_config(config_yaml, LightningPolicyConfig)
    action_mapping = UniversalAutoregressiveActionMapping(config=config.shared.action_mapping)

    # 与官方 InferenceServer 一致：用 pl.Trainer.init_module 把模型放到 GPU + 正确精度
    dummy_trainer = pl.Trainer(precision=config.shared.precision, accelerator="gpu", devices=[0])
    with dummy_trainer.init_module():
        model = Stage3LabelledBCLightning.load_from_checkpoint(
            args.weights, config=config, inference_mode=True
        )
    model.eval()

    # ---- 评测口径：teacher_forcing（一次前向，GT 历史）----
    if args.mode == "teacher_forcing":
        preds = _predict_teacher_forcing(model, config, action_mapping, frames_np, samples)
        _write_predictions(args, samples, preds, n_frames)
        return 0

    # ---- 演示口径：free_running（重建官方 FullInferenceState 的等价喂帧循环）----
    T = config.shared.n_seq_timesteps  # 200
    n_actions = action_mapping.get_seq_len()  # max_keys(4) + max_mouse_keys(2) + 2 = 8
    n_key = action_mapping.get_number_of_keyboard_actions()  # 4
    n_mouse_btn = action_mapping.get_number_of_mouse_button_actions()  # 2
    text_embed_dim = model._get_text_embedding_dim()  # 768（gemma）
    device = model.device

    frame_in = torch.zeros((1, T, 3, 192, 192), dtype=torch.uint8, device=device)
    action_in = action_mapping.make_empty_action(T)  # 各字段 (T, n)，在 cpu
    action_in = pytree.tree_map(lambda x: x.unsqueeze(0).to(device), action_in)  # (1,T,n)
    text_tokens_embed = torch.zeros(
        (1, T, 1, text_embed_dim), dtype=torch.bfloat16, device=device
    )

    n_prior = 0
    preds = []
    for i in range(n_frames):
        frame = torch.from_numpy(frames_np[i]).to(device)  # (3,192,192) uint8
        if n_prior < T:
            frame_in[:, n_prior, :, :, :] = frame
            n_prior += 1
        else:
            frame_in = torch.roll(frame_in, -1, dims=1)
            action_in = pytree.tree_map(lambda x: torch.roll(x, -1, dims=1), action_in)
            text_tokens_embed = torch.roll(text_tokens_embed, -1, dims=1)
            frame_in[:, -1, :, :, :] = frame
            text_tokens_embed[:, -1, :, :] = 0

        # 自回归解码：n_actions 次前向，每次填当前帧动作的一个 token
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for j in range(n_actions):
                action, _ = model.online_full_predict(
                    frames=frame_in,
                    actions=action_in,
                    sampling_temperature=config.inference.sampling_temperature,
                    text_tokens_embed=text_tokens_embed,
                )
                pos = n_prior - 1
                if j < n_key:
                    action_in.keys[:, pos, j] = action.keys[:, pos, j]
                elif j < n_key + n_mouse_btn:
                    action_in.mouse_buttons[:, pos, j - n_key] = action.mouse_buttons[:, pos, j - n_key]
                elif j < n_key + n_mouse_btn + 1:
                    action_in.mouse_delta_x[:, pos, 0] = action.mouse_delta_x[:, pos, 0]
                else:
                    action_in.mouse_delta_y[:, pos, 0] = action.mouse_delta_y[:, pos, 0]

        sampled_action = pytree.tree_map(lambda x: x[:, n_prior - 1, :], action_in)
        act = action_mapping.tensor_to_action(
            sampled_action, mouse_sampling_approach=config.inference.mouse_sampling_approach
        )
        preds.append({
            "frame_id": samples["frames"][i]["frame_id"],
            "keys": list(act.keys),
            "mouse_buttons": list(act.mouse_buttons),
            "mouse_delta_x": int(act.mouse_delta_x),
            "mouse_delta_y": int(act.mouse_delta_y),
        })
        if (i + 1) % 20 == 0 or i + 1 == n_frames:
            print(f"[进度] {i + 1}/{n_frames} 帧", file=sys.stderr, flush=True)

    _write_predictions(args, samples, preds, n_frames)
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))

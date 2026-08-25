#!/usr/bin/env python3
"""smoke_test.py —— 快速通检（第 7 天 · 端到端联调冒烟测试）。

按第 7 天要求：一小套用例确认主路径没有当场失效——不少于 5 条、含至少 1 条异常路径，
主路径条目对上第 5 天工作流程的步骤编号（步骤 1 preprocess / 2 infer teacher_forcing /
3 evaluate / 4 infer free_running 演示）。每条用例核对退出码（0 成功 / 2 参数错 / 3 数据错）
+ 期望输出片段，最后打一张「编号 / 步骤 / 期望 / 实际 / 是否通过」的通检表。

环境自适应（同一脚本本地与服务器都能跑）：
- 无 torchcodec → 步骤 1 跳过（读帧需 torchcodec，本机未装）；
- 无 CUDA GPU → 步骤 2/4 跳过（150M 为 CUDA 权重，本机 CPU 跑不动）；
- 步骤 3（evaluate）纯标准库，本机也可跑：无冒烟产物时回退到 golden 产物回归 88%/0.756。

本脚本自身退出码：0=无致命失败（通过或仅环境跳过）／2=有致命用例失败。
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable  # 用启动本脚本的同一个解释器跑三脚本，继承 venv


def parse_args(argv):
    p = argparse.ArgumentParser(description="快速通检（端到端联调冒烟测试）")
    p.add_argument("--data_dir", default="data/p2p-toy-examples", help="toy 数据目录（服务器默认名）")
    p.add_argument("--weights", default="checkpoints/150M/checkpoint-step=00500000.ckpt", help="150M 权重")
    p.add_argument("--samples", default="out/samples.json", help="golden 样本（本机回归用）")
    p.add_argument("--pred", default="pred/predictions.json", help="golden 预测（本机回归用）")
    p.add_argument("--official-repo", default=None, help="官方 open-p2p 仓库根目录（含 elefant）")
    p.add_argument("--smoke_out", default="out/smoke", help="冒烟 preprocess 输出目录")
    p.add_argument("--smoke_pred", default="pred/smoke", help="冒烟 infer 输出目录")
    return p.parse_args(argv)


def _can_gpu():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _can_torchcodec():
    try:
        import torchcodec  # noqa: F401
        return True
    except Exception:
        return False


def _run(cmd):
    """跑一条命令，返回 (returncode, 合并的 stdout+stderr 文本)。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode, out


def main(argv):
    args = parse_args(argv)
    gpu = _can_gpu()
    codec = _can_torchcodec()
    print(f"[smoke] 环境探测：CUDA GPU={'是' if gpu else '否'}，torchcodec={'是' if codec else '否'}")
    print(f"[smoke] 解释器：{PY}\n")

    rows = []  # (编号, 步骤, 期望, 实际, 是否通过)

    def add(num, step, expect, actual, passed):
        rows.append((num, step, expect, actual, "通过" if passed else "失败"))

    # ---- 主路径（端到端，写冒烟隔离目录，不对号 golden 产物）----
    # 步骤 1：preprocess
    if not codec:
        rows.append(("1", "步骤1 preprocess", "exit 0 + [OK] 处理 200 帧",
                     "跳过（本机无 torchcodec，需 GPU 服务器）", None))
    else:
        cmd = [PY, "scripts/preprocess.py", "--data_dir", args.data_dir,
               "--out_dir", args.smoke_out, "--n_frames", "200"]
        rc, out = _run(cmd)
        ok = rc == 0 and "[OK] 处理" in out
        add("1", "步骤1 preprocess", "exit 0 + [OK] 处理 200 帧", f"exit {rc}", ok)

    # 步骤 2：infer teacher_forcing（依赖步骤 1 产物）
    samples_smoke = os.path.join(args.smoke_out, "samples.json")
    if not gpu:
        rows.append(("2", "步骤2 infer(teacher_forcing)", "exit 0 + [OK] 推理 200 帧完成",
                     "跳过（本机无 CUDA GPU）", None))
    elif not os.path.isfile(samples_smoke):
        rows.append(("2", "步骤2 infer(teacher_forcing)", "exit 0 + [OK] 推理 200 帧完成",
                     f"失败（缺 {samples_smoke}，步骤1 未产出）", False))
    else:
        cmd = [PY, "scripts/infer.py", "--weights", args.weights, "--samples", samples_smoke,
               "--out", args.smoke_pred, "--mode", "teacher_forcing"]
        if args.official_repo:
            cmd += ["--official-repo", args.official_repo]
        rc, out = _run(cmd)
        ok = rc == 0 and "[OK] 推理" in out
        add("2", "步骤2 infer(teacher_forcing)", "exit 0 + [OK] 推理 200 帧完成", f"exit {rc}", ok)

    # 步骤 3：evaluate（纯标准库，处处可跑；优先冒烟产物，缺则回退 golden 回归）
    pred_smoke = os.path.join(args.smoke_pred, "predictions.json")
    use_pred, use_label = pred_smoke, samples_smoke
    if not (os.path.isfile(pred_smoke) and os.path.isfile(samples_smoke)):
        use_pred, use_label = args.pred, args.samples
    cmd = [PY, "scripts/evaluate.py", "--pred", use_pred, "--label", use_label,
           "--out", "metrics.smoke.json"]
    rc, out = _run(cmd)
    ok = rc == 0 and "按键准确率" in out
    src = "冒烟产物" if use_pred == pred_smoke else "golden 产物回归"
    add("3", f"步骤3 evaluate（{src}）", "exit 0 + [OK] 按键准确率", f"exit {rc}（{src}）", ok)

    # 步骤 4：infer free_running（演示口径）
    if not gpu:
        rows.append(("4", "步骤4 infer(free_running)", "exit 0 + [OK] 推理 200 帧完成",
                     "跳过（本机无 CUDA GPU）", None))
    else:
        cmd = [PY, "scripts/infer.py", "--weights", args.weights, "--samples", samples_smoke,
               "--out", args.smoke_pred + "_free", "--mode", "free_running"]
        if args.official_repo:
            cmd += ["--official-repo", args.official_repo]
        rc, out = _run(cmd)
        ok = rc == 0 and "[OK] 推理" in out
        add("4", "步骤4 infer(free_running)", "exit 0 + [OK] 推理 200 帧完成", f"exit {rc}", ok)

    # ---- 异常路径（处处可跑，≥1 条）----
    cmd = [PY, "scripts/preprocess.py", "--data_dir", args.data_dir, "--out_dir", "out/smoke", "--n_frames", "-5"]
    rc, _ = _run(cmd)
    add("5", "异常 preprocess --n_frames -5", "exit 2 + [ERROR]", f"exit {rc}", rc == 2)

    cmd = [PY, "scripts/preprocess.py", "--data_dir", "/nonexistent", "--out_dir", "out/smoke", "--n_frames", "200"]
    rc, _ = _run(cmd)
    add("6", "异常 preprocess 数据目录不存在", "exit 3 + [ERROR]", f"exit {rc}", rc == 3)

    cmd = [PY, "scripts/infer.py", "--weights", "/nonexistent.ckpt", "--samples", args.samples, "--out", "pred/smoke"]
    rc, _ = _run(cmd)
    add("7", "异常 infer 权重不存在", "exit 3 + [ERROR] 权重文件不存在", f"exit {rc}", rc == 3)

    cmd = [PY, "scripts/evaluate.py", "--pred", "/nonexistent", "--label", "/nonexistent"]
    rc, _ = _run(cmd)
    add("8", "异常 evaluate 预测/标注不存在", "exit 3 + [ERROR]", f"exit {rc}", rc == 3)

    # ---- 通检表 ----
    print("\n" + "=" * 78)
    print("通检表（编号 / 步骤 / 期望 / 实际 / 是否通过）")
    print("=" * 78)
    for num, step, expect, actual, passed in rows:
        mark = passed if passed in ("通过", "失败") else "—"
        print(f"#{num}  {step}\n    期望: {expect}\n    实际: {actual}\n    结论: {mark}")
    fatal = any(p == "失败" for _, _, _, _, p in rows)
    print("=" * 78)
    print(f"汇总：{sum(1 for r in rows if r[4] == '通过')} 通过 / "
          f"{sum(1 for r in rows if r[4] == '失败')} 失败 / "
          f"{sum(1 for r in rows if r[4] is None)} 跳过（环境缺失）")
    return 2 if fatal else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
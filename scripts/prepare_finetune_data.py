#!/usr/bin/env python3
"""prepare_finetune_data.py —— 微调数据准备（块 2，服务器侧，扩展）。

从 full-data 下载“指定 batch 子集”（不碰全库），按 UUID→游戏 过滤出目标游戏、
排除测试段 UUID、按帧数截取 N 小时，产出微调数据目录（供 train.py 使用）。

为什么需要这个脚本：官方 `scripts/download_data.py` 只做“按 batch 范围下载 + 解压”，
没有按游戏过滤、也没有排除测试段、截取固定时长——而微调要求“1 款游戏、2 小时子集”，
且必须把测试段（toy 第一段 = call-of-duty-mobile，UUID 01993060-…）从训练数据里剔除，
否则训练/测试数据泄漏、指标虚高。所以这一步要单独落地（对齐“诚实优先”）。

关键口径（出处见 项目备忘.md / 开发日志.md 第 6 天）：
- full-data 是 545 个 batch（`dataset/batch_00001~00545.tar.gz`，每批 ~200 条），
  batch 按 UUIDv7 时间序分块；batch→游戏 映射靠 `data_metadata.parquet`（UUID→env_name+num_frames）。
- call-of-duty-mobile 最密集在 batch 167（45 条 / 3.1h），单批即够 2h，故默认下 166~168 兜底。
- 测试段 UUID 默认排除 `01993060-06a0-7352-ac0b-d77d50a1aba7`（batch 317）。
- 帧率固定 20fps（官方 annotation.frames_per_second），2h = 144000 帧。

依赖：huggingface_hub（下载）+ pyarrow（读 parquet）。仅在服务器（有网络 + 磁盘）运行；
本机 Windows 无 N 卡、也无需下这批数据。服务器直连 huggingface.co 被墙（国内网络），
须设 HF_ENDPOINT=https://hf-mirror.com（国内镜像）再跑本脚本，否则下载会卡死。

退出码 0=成功 / 2=参数错误 / 3=数据错误。约定见 接口约定.md。
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import uuid as _uuid
from pathlib import Path

FPS = 20  # 官方 annotation.frames_per_second


def parse_args(argv):
    p = argparse.ArgumentParser(description="full-data batch 子集 → 1 款游戏 N 小时微调数据目录")
    p.add_argument("--repo", default="elefantai/p2p-full-data",
                   help="full-data 数据集仓库（默认 elefantai/p2p-full-data）")
    p.add_argument("--start", type=int, default=166, help="起始 batch（含，默认 166）")
    p.add_argument("--end", type=int, default=168, help="结束 batch（含，默认 168，即 COD 密集 batch 167 ± 1 兜底）")
    p.add_argument("--game", default="call-of-duty-mobile", help="目标游戏 env_name（默认 call-of-duty-mobile）")
    p.add_argument("--hours", type=float, default=2.0, help="截取时长（小时，默认 2.0）")
    p.add_argument("--exclude", default="01993060-06a0-7352-ac0b-d77d50a1aba7",
                   help="逗号分隔的要排除的 recording UUID（默认排除测试段，防泄漏）")
    p.add_argument("--out_dir", default="data/codm-2h", help="输出微调数据目录（默认 data/codm-2h）")
    p.add_argument("--work_dir", default="_ft_download", help="临时下载/解压目录（默认 _ft_download，跑完删除）")
    p.add_argument("--keep_work", action="store_true", help="保留 work_dir（不删除临时下载）")
    p.add_argument("--skip_download", action="store_true",
                   help="跳过 snapshot_download（tar.gz 与 data_metadata.parquet 已手动放到 work_dir，如经 curl 断点续传）")
    return p.parse_args(argv)


def validate_args(args):
    if args.start < 1 or args.end < args.start:
        print(f"[ERROR] --start/--end 非法：{args.start}~{args.end}（退出码 2）", file=sys.stderr)
        return 2
    if args.hours <= 0:
        print(f"[ERROR] --hours 必须为正，收到 {args.hours}（退出码 2）", file=sys.stderr)
        return 2
    return 0


def _to_uuid_str(v):
    """把 parquet 里的 UUID（bytes / uuid.UUID / str）统一成小写标准字符串。"""
    if isinstance(v, bytes):
        try:
            return str(_uuid.UUID(bytes=v))
        except ValueError:
            return v.decode("utf-8", "ignore").lower()
    if isinstance(v, _uuid.UUID):
        return str(v)
    return str(v).strip().lower()


def load_game_map(metadata_path):
    """读 data_metadata.parquet → {uuid_str: (env_name, num_frames)}。

    目录名到底对 `id`（二进制 UUIDv7）还是 `filepath`（字符串 UUID）不 100% 确定，
    故两列都建 key（同一条 recording 的 id/filepath 若不同，各自都指向同一 game/frames），
    解压目录名对上哪个都能命中。
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"[ERROR] 需要 pyarrow 读 parquet: {e}（退出码 3）", file=sys.stderr)
        return None

    table = pq.read_table(metadata_path)
    names = set(table.column_names)
    if "env_name" not in names or "num_frames" not in names:
        print(f"[ERROR] parquet 缺 env_name/num_frames 列，实际列：{sorted(names)}（退出码 3）", file=sys.stderr)
        return None
    id_cols = [c for c in ("id", "filepath") if c in names]
    if not id_cols:
        print(f"[ERROR] parquet 缺 UUID 列（id/filepath），实际列：{sorted(names)}（退出码 3）", file=sys.stderr)
        return None

    games = table.column("env_name").to_pylist()
    frames = table.column("num_frames").to_pylist()

    game_map = {}
    for row_idx in range(len(games)):
        val = (str(games[row_idx]), int(frames[row_idx]))
        for col in id_cols:
            key = _to_uuid_str(table.column(col)[row_idx].as_py())
            game_map[key] = val
    print(f"[OK] 读 parquet：{metadata_path}，{len(game_map)} 个 UUID key（列 {id_cols}）")
    return game_map


def download(repo, batches, metadata_path, work_dir):
    """下载 parquet + 指定 batch 的 tar.gz 到 work_dir（用官方同款 snapshot_download 签名）。"""
    from huggingface_hub import snapshot_download
    os.makedirs(work_dir, exist_ok=True)

    # 1) parquet（6MB，无论它在仓库哪个子目录都能命中）
    print(f"[OK] 下载 data_metadata.parquet → {work_dir}")
    snapshot_download(repo, repo_type="dataset",
                      allow_patterns=["*data_metadata.parquet*"], local_dir=work_dir)
    parquet_files = list(Path(work_dir).rglob("data_metadata.parquet"))
    if not parquet_files:
        print("[ERROR] 下载后未找到 data_metadata.parquet（退出码 3）", file=sys.stderr)
        return None
    src = parquet_files[0]
    if src.resolve() != Path(metadata_path).resolve():
        shutil.copy2(src, metadata_path)

    # 2) batch 打包（每个 ~200 条混合游戏）
    patterns = [f"*batch_{i:05d}*" for i in batches]
    print(f"[OK] 下载 batch {batches[0]}~{batches[-1]}（{len(batches)} 个 tar.gz）→ {work_dir}")
    snapshot_download(repo, repo_type="dataset", allow_patterns=patterns, local_dir=work_dir)
    return metadata_path


def extract_all(work_dir, extract_dir):
    """解压 work_dir 下所有 *.tar.gz 到 extract_dir，删掉 tar。

    只取训练/评测需要的 `192x192.mp4` + `annotation.proto`，跳过冗余原片 `video.mp4`
    （toy 数据已证实每个 recording 都带一份高清原片，微调/推理流水线用不到，跳过省 ~90% 磁盘）。
    """
    os.makedirs(extract_dir, exist_ok=True)
    tars = sorted(Path(work_dir).rglob("*.tar.gz"))
    if not tars:
        print(f"[ERROR] 未找到 tar.gz（{work_dir}）（退出码 3）", file=sys.stderr)
        return False
    keep = ("192x192.mp4", "annotation.proto")
    for tar in tars:
        print(f"[OK] 解压 {tar.name}（只取 192x192.mp4 + annotation.proto）")
        with tarfile.open(tar, "r:gz") as tf:
            members = [m for m in tf.getmembers()
                       if m.isfile() and m.name.endswith(keep)]
            tf.extractall(extract_dir, members=members)
        tar.unlink()
    return True


def select_recordings(extract_dir, game_map, game, exclude, target_frames):
    """扫描 extract_dir 下每个 <uuid>/annotation.proto，过滤出目标游戏、排除指定 UUID、
    按 UUID（时间序）累计帧数截到 target_frames，返回选中列表 [{uuid, num_frames}]。"""
    selected = []
    for proto in sorted(Path(extract_dir).rglob("annotation.proto")):
        rec_dir = proto.parent
        rec_uuid = rec_dir.name
        info = game_map.get(rec_uuid.lower())
        if info is None:
            continue  # parquet 里查不到（不应发生，跳过并继续）
        g, nframes = info
        if g != game:
            continue
        if rec_uuid.lower() in exclude:
            print(f"[OK] 排除测试/指定 recording：{rec_uuid}")
            continue
        selected.append((rec_uuid, rec_dir, nframes))

    # UUIDv7 时间序 = 字符串排序即时间序
    selected.sort(key=lambda t: t[0])
    picked, total = [], 0
    for rec_uuid, rec_dir, nframes in selected:
        picked.append({"uuid": rec_uuid, "num_frames": nframes})
        total += nframes
        if total >= target_frames:
            break

    if not picked:
        print(f"[ERROR] 过滤后无 recording（game={game}），检查 batch 范围/游戏名（退出码 3）", file=sys.stderr)
        return None
    return picked, total


def run(args):
    err = validate_args(args)
    if err:
        return err

    work_dir = Path(args.work_dir)
    metadata_path = work_dir / "data_metadata.parquet"
    extract_dir = work_dir / "extracted"
    out_dir = Path(args.out_dir)
    batches = list(range(args.start, args.end + 1))
    exclude = {s.strip().lower() for s in args.exclude.split(",") if s.strip()}
    target_frames = int(args.hours * 3600 * FPS)

    # 1) 下载 + 解压
    if args.skip_download:
        if not metadata_path.is_file():
            print(f"[ERROR] --skip_download 除 tar.gz 外还需 {metadata_path}（data_metadata.parquet，6MB）也在 work_dir；"
                  f"只下 tar.gz 不够，请一并放好或去掉 --skip_download 走下载（退出码 3）", file=sys.stderr)
            return 3
        print(f"[OK] --skip_download：跳过下载，直接用 {work_dir} 下已有的 tar.gz/parquet")
    elif not download(args.repo, batches, str(metadata_path), str(work_dir)):
        return 3
    if not extract_all(str(work_dir), str(extract_dir)):
        return 3

    # 2) 过滤 + 截取
    game_map = load_game_map(str(metadata_path))
    if game_map is None:
        return 3
    result = select_recordings(str(extract_dir), game_map, args.game, exclude, target_frames)
    if result is None:
        return 3
    picked, total = result
    print(f"[OK] 选中 {len(picked)} 条 {args.game} recording，共 {total} 帧 = {total / (3600 * FPS):.2f}h（目标 ~{args.hours}h / {target_frames} 帧）")

    # 3) 搬到 out_dir + 写 manifest
    os.makedirs(out_dir, exist_ok=True)
    for item in picked:
        src = extract_dir / item["uuid"]
        shutil.move(str(src), str(out_dir / item["uuid"]))
    manifest = {
        "repo": args.repo,
        "game": args.game,
        "batches": batches,
        "hours": args.hours,
        "target_frames": target_frames,
        "exclude": sorted(exclude),
        "selected": picked,
        "total_frames": total,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 4) 清理临时目录
    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"[OK] 清理临时目录 {work_dir}")

    print(f"[OK] 微调数据就绪：{out_dir}（{len(picked)} 条，manifest.json 记录明细）")
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))
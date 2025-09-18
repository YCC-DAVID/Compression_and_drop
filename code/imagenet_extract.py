#!/usr/bin/env python3
"""
ImageNet (ILSVRC2012) extractor

功能：
1) 解压训练集 ILSVRC2012_img_train.tar
   - 该 tar 里包含 1000 个以 wnid 命名的子 tar（如 n01440764.tar）。
   - 本脚本会先解出这些子 tar，再分别解到 train/<wnid>/ 目录中。
2) 解压验证集 ILSVRC2012_img_val.tar 到 val/
3) （可选）用 devkit 里的 meta.mat 与 ILSVRC2012_validation_ground_truth.txt
   把验证集图片重新按类别（wnid）归类到 val/<wnid>/。

依赖：
- Python 3.8+
- 标准库：tarfile, argparse, shutil, tempfile, pathlib, multiprocessing, re, json
- 可选：tqdm (用于进度条)，scipy (用于读取 devkit 的 meta.mat)

示例：
python imagenet_extract.py \
  --src /path/to/ILSVRC2012 \
  --out /path/to/imagenet \
  --do-train --do-val --devkit

目录期待：
/src/ILSVRC2012_img_train.tar
/src/ILSVRC2012_img_val.tar
/src/ILSVRC2012_devkit_t12.tar.gz  (当使用 --devkit 时)

输出：
/out/train/<wnid>/*.JPEG
/out/val/<wnid>/*.JPEG  （如果使用 --devkit，会自动按类分到子目录）

注意：
- 解压过程会占用较多磁盘空间；请确保 /out 至少有 ~250GB 空间。
- Windows 下请避免过深的路径（260+字符限制）。
"""
from __future__ import annotations
import argparse
import tarfile
from pathlib import Path
import tempfile
import shutil
import multiprocessing as mp
from typing import Iterable, Tuple, Dict, List
import re

try:
    from tqdm import tqdm  # type: ignore
    TQDM = True
except Exception:
    TQDM = False


def _extract_tar(tar_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:*') as tf:
        def is_within_directory(directory, target):
            abs_directory = Path(directory).resolve()
            abs_target = Path(target).resolve()
            try:
                abs_target.relative_to(abs_directory)
                return True
            except ValueError:
                return False
        # 安全解压，防止目录穿越
        for member in tf.getmembers():
            member_path = dest / member.name
            if not is_within_directory(dest, member_path):
                raise Exception(f"Blocked path traversal in tar file: {member.name}")
        tf.extractall(dest)


def _extract_single_class_tar(args: Tuple[Path, Path]):
    class_tar, out_root = args
    wnid = class_tar.stem  # e.g., n01440764
    out_dir = out_root / wnid
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(class_tar, 'r:*') as tf:
        tf.extractall(out_dir)


def extract_train(src: Path, out: Path, workers: int = 8) -> None:
    train_tar = src / 'ILSVRC2012_img_train.tar'
    if not train_tar.exists():
        raise FileNotFoundError(f"找不到 {train_tar}")

    train_root = out / 'train'
    train_root.mkdir(parents=True, exist_ok=True)

    # 先把 1000 个子 tar 解出来
    with tempfile.TemporaryDirectory() as tmpd:
        tmpd = Path(tmpd)
        print(f"[train] 解出子 tar 到临时目录: {tmpd}")
        _extract_tar(train_tar, tmpd)
        class_tars = sorted(tmpd.glob('*.tar'))
        if len(class_tars) != 1000:
            print(f"[警告] 期望 1000 个子 tar，实际发现 {len(class_tars)} 个。继续...")

        iterable: Iterable = class_tars
        if TQDM:
            iterable = tqdm(class_tars, desc='[train] 解压每个类别')

        # 多进程解每个类别 tar
        if workers > 1:
            with mp.Pool(processes=workers) as pool:
                pool.map(_extract_single_class_tar, [(p, train_root) for p in class_tars])
        else:
            for p in iterable:
                _extract_single_class_tar((p, train_root))

    print("[train] 训练集解压完成。")


def extract_val(src: Path, out: Path) -> None:
    val_tar = src / 'ILSVRC2012_img_val.tar'
    if not val_tar.exists():
        raise FileNotFoundError(f"找不到 {val_tar}")

    val_root = out / 'val'
    val_root.mkdir(parents=True, exist_ok=True)

    print("[val] 解压验证集图片到 val 根目录……")
    _extract_tar(val_tar, val_root)
    # 大多数情况下，解出的是一堆 JPEG 到 val 根目录

    print("[val] 验证集解压完成。")


# ---- devkit 支持：把 val 图片分到对应 wnid 目录 ----

def _load_meta_wnids_from_devkit(devkit_dir: Path) -> Dict[int, str]:
    """返回 {ILSVRC2012_ID (1..1000): wnid} 的映射。
    需要 scipy.io.loadmat 读取 meta.mat。
    """
    data_dir = devkit_dir / 'data'
    meta_mat = data_dir / 'meta.mat'
    if not meta_mat.exists():
        raise FileNotFoundError(f"devkit 中缺少 {meta_mat}")

    try:
        from scipy.io import loadmat  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "需要 scipy 才能解析 devkit 的 meta.mat。请先安装：pip install scipy"
        ) from e

    mat = loadmat(str(meta_mat))
    synsets = mat.get('synsets')
    if synsets is None:
        raise RuntimeError("meta.mat 中未找到 'synsets'")

    # synsets 是结构体数组，字段里有 ILSVRC2012_ID 和 WNID（大小写可能因版本不同略有差异，这里做兜底）
    # 兼容字段名
    def get_field(arr, idx, name_candidates: List[str]):
        for n in name_candidates:
            try:
                return arr[idx][n][0]
            except Exception:
                pass
        raise KeyError(f"字段 {name_candidates} 未找到")

    id2wnid: Dict[int, str] = {}
    # synsets 的形状通常是 (1000, 1)
    N = synsets.shape[0]
    for i in range(N):
        try:
            ilsvrc_id = int(get_field(synsets, i, ['ILSVRC2012_ID', 'ILSVRC2012_ID'][0]))
        except Exception:
            # 有些加载结果需要不同的下标访问方式
            try:
                ilsvrc_id = int(synsets[i][0]['ILSVRC2012_ID'][0][0])
            except Exception as e:
                raise RuntimeError("无法从 meta.mat 解析 ILSVRC2012_ID") from e
        try:
            wnid_raw = get_field(synsets, i, ['WNID', 'wnid'])
            wnid = str(wnid_raw[0]) if isinstance(wnid_raw, (list, tuple)) else str(wnid_raw)
        except Exception:
            try:
                wnid = str(synsets[i][0]['WNID'][0])
            except Exception as e:
                raise RuntimeError("无法从 meta.mat 解析 WNID") from e
        if 1 <= ilsvrc_id <= 1000:
            id2wnid[ilsvrc_id] = wnid

    if len(id2wnid) != 1000:
        print(f"[警告] 从 devkit 解析到 {len(id2wnid)} 个 wnid（非 1000）。继续……")
    return id2wnid


def rearrange_val_with_devkit(src: Path, out: Path) -> None:
    devkit_tar = src / 'ILSVRC2012_devkit_t12.tar.gz'
    if not devkit_tar.exists():
        raise FileNotFoundError(f"找不到 {devkit_tar}")

    with tempfile.TemporaryDirectory() as tmpd:
        tmpd = Path(tmpd)
        print(f"[devkit] 解压 devkit 到临时目录: {tmpd}")
        _extract_tar(devkit_tar, tmpd)

        # devkit 解出后一般是 ILSVRC2012_devkit_t12/ 目录
        # 兼容择其一
        candidates = list(tmpd.glob('ILSVRC2012_devkit_t12')) + list(tmpd.glob('ILSVRC2012_devkit'))
        if not candidates:
            raise RuntimeError("未在 devkit 压缩包中找到 ILSVRC2012_devkit_t12 目录")
        devkit_dir = candidates[0]

        # 读取 id->wnid 映射
        id2wnid = _load_meta_wnids_from_devkit(devkit_dir)

        # 读取验证集真值（按文件号顺序给出 1..50000 的类别 id）
        gt_txt = devkit_dir / 'data' / 'ILSVRC2012_validation_ground_truth.txt'
        if not gt_txt.exists():
            raise FileNotFoundError(f"缺少 {gt_txt}")
        gt = [int(x.strip()) for x in gt_txt.read_text().splitlines() if x.strip()]
        if len(gt) != 50000:
            print(f"[警告] 验证集真值条目数 = {len(gt)}（非 50000）。继续……")

    # 将 val 根目录下的图片移动到对应 wnid 目录
    val_root = out / 'val'
    imgs = sorted(val_root.glob('*.JPEG'))
    # 文件名形如 ILSVRC2012_val_00000001.JPEG
    pat = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$", re.IGNORECASE)
    moved = 0
    iterable = imgs
    if TQDM:
        iterable = tqdm(imgs, desc='[val] 归类验证集')
    for img in iterable:
        m = pat.search(img.name)
        if not m:
            continue
        idx = int(m.group(1))  # 1-based
        if 1 <= idx <= len(gt):
            class_id = gt[idx - 1]
            wnid = id2wnid.get(class_id)
            if wnid is None:
                continue
            dest_dir = val_root / wnid
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img), str(dest_dir / img.name))
            moved += 1
    print(f"[val] 已移动 {moved} 张验证集图片到各类别目录。")


def parse_args():
    p = argparse.ArgumentParser(description='ImageNet (ILSVRC2012) 解压与整理脚本')
    p.add_argument('--src', type=Path, required=True, help='包含原始 tar 的目录')
    p.add_argument('--out', type=Path, required=True, help='输出根目录')
    p.add_argument('--do-train', action='store_true', help='处理训练集')
    p.add_argument('--do-val', action='store_true', help='处理验证集')
    p.add_argument('--devkit', action='store_true', help='使用 devkit 把验证集按 wnid 归类')
    p.add_argument('--workers', type=int, default=max(1, mp.cpu_count() // 2), help='解压训练集类别 tar 的并行进程数')
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.do_train:
        extract_train(args.src, args.out, workers=args.workers)

    if args.do_val:
        extract_val(args.src, args.out)
        if args.devkit:
            rearrange_val_with_devkit(args.src, args.out)

    if not (args.do_train or args.do_val):
        print("未选择 --do-train 或 --do-val，未执行任何操作。")


if __name__ == '__main__':
    main()

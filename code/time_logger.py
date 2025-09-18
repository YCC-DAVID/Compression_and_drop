# logger_json.py
import time
import torch
import csv
from torchvision import transforms

import os, json
from collections import defaultdict

class BatchTableLogger:
    """
    一行=一个batch。列：
      epoch, iter, decompress_avg, transfer, forward, backward, aug.<Name1>, aug.<Name2>, ...
    augmentation 列为该 batch 内“该增强的平均耗时”（sum/count）。
    每个 epoch 结束时写出一个 CSV 文件（列头包含本 epoch 出现过的全部增强名）。
    """
    def __init__(self, log_dir="timing_logs", 
                 filename_tpl="batches_epoch{epoch}.csv", 
                 sep=",", 
                 ndigits=6,
                 clean_on_init: bool = False,          # <<< 新增：是否在初始化时清理
                 clean_mode: str = "template",         # "template" | "all"
                 extra_patterns: tuple[str, ...] = (), # 额外要清理的通配符
                 dry_run: bool = False):                # 先看看会删哪些（不真正删除）):
        
        os.makedirs(log_dir, exist_ok=True)

        self.log_dir, self.filename_tpl, self.sep, self.ndigits = log_dir, filename_tpl, sep, ndigits
        self.epoch = None
        self._rows = []                 # 本 epoch 的所有 batch 行
        self._seen_aug = set()          # 本 epoch 出现过的所有增强名

        # —— 当前 batch 的累加器（sum 与 count）——
        self._aug_sum = defaultdict(float)
        self._aug_cnt = defaultdict(int)
        self._decomp_sum = 0.0
        self._decomp_cnt = 0
        self._transfer = 0.0
        self._forward = 0.0
        self._backward = 0.0
        self._load_time = 0.0

    # ===== 生命周期 =====
    def start_epoch(self, epoch:int):
        self.epoch = epoch
        self._rows.clear()
        self._seen_aug.clear()
        self.reset_iter()

    def end_epoch_write(self):
        if self.epoch is None: return
        cols = ["epoch","iter","decompress_avg","decompress_sum","load_time","transfer","forward","backward"] \
             + [f"aug.{n}" for n in sorted(self._seen_aug)]+ [f"aug.{n}_sum" for n in sorted(self._seen_aug)]
        path = os.path.join(self.log_dir, self.filename_tpl.format(epoch=self.epoch))
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter=self.sep)
            w.writerow(cols)
            for r in self._rows:
                row = [
                    r["epoch"], r["iter"],
                    self._fmt(r.get("decompress_avg", 0.0)),
                    self._fmt(r.get("decompress_sum", 0.0)),
                    self._fmt(r.get("load_time", 0.0)),
                    self._fmt(r.get("transfer", 0.0)),
                    self._fmt(r.get("forward", 0.0)),
                    self._fmt(r.get("backward", 0.0)),
                ]
                for n in sorted(self._seen_aug):
                    row.append(self._fmt(r["aug_avg"].get(n, 0.0)))
                    row.append(self._fmt(r["aug_sum"].get(n, 0.0)))
                w.writerow(row)

    def reset_iter(self):
        self._aug_sum.clear(); self._aug_cnt.clear()
        self._decomp_sum = 0.0; self._decomp_cnt = 0
        self._transfer = self._forward = self._backward = self._load_time = 0.0

    # ===== 在 batch 内累加 =====
    def acc_aug(self, name:str, t:float):
        self._aug_sum[name] += float(t)
        self._aug_cnt[name] += 1
        self._seen_aug.add(name)

    def acc_decompress(self, t:float):
        self._decomp_sum += float(t); self._decomp_cnt += 1

    def set_transfer(self, t:float): self._transfer = float(t)
    def set_forward(self, t:float):  self._forward  = float(t)
    def set_backward(self, t:float): self._backward = float(t)
    def set_load_time(self, t:float): self._load_time = float(t)

    # ===== 结束一个 batch，写入一行（缓存到内存；epoch 末统一落盘）=====
    def commit_iter_row(self, it:int):
        decomp_avg = (self._decomp_sum / self._decomp_cnt) if self._decomp_cnt else 0.0
        decomp_sum = self._decomp_sum
        names = set(self._aug_sum) | set(self._aug_cnt)
        aug_avg = {
        n: (self._aug_sum.get(n, 0.0) / self._aug_cnt.get(n, 0)) if self._aug_cnt.get(n, 0) else 0.0
        for n in names}
        aug_sum = {n: self._aug_sum[n] for n in self._aug_sum.keys()}
        self._rows.append({
            "epoch": self.epoch, "iter": it,
            "decompress_avg": decomp_avg,
            "decompress_sum": decomp_sum,
            "load_time": self._load_time,
            "transfer": self._transfer,
            "forward": self._forward,
            "backward": self._backward,
            "aug_avg": aug_avg,
            "aug_sum": aug_sum,
        })
        self.reset_iter()

    # ===== 辅助 =====
    def _fmt(self, x:float): return f"{x:.{self.ndigits}f}"

class TimingContext:
    current_epoch = 0
    current_iter = 0
    logger = None

class TimedTransform:
    def __init__(self, transform, name=None, cuda_sync=False):
        self.transform = transform
        self.name = name or transform.__class__.__name__
        self.cuda_sync = cuda_sync

    def __call__(self, *args, **kwargs):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.transform(*args, **kwargs)
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if TimingContext.logger is not None:
            TimingContext.logger.acc_aug(self.name, dt)  # 批内累加
        return out

def timed_compose(ts):
    wrapped = []
    for t in ts:
        if t is None: continue
        wrapped.append(TimedTransform(t))
    return transforms.Compose(wrapped)

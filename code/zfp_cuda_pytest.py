import time
import math
import statistics as stats
import torch
import numpy as np

# ======== 配置 ========
device = "cuda"
dtype = torch.float32
tolerances = [1e0, 1e1, 1e2]     # 可按需修改
repeats = 10
warmup = 3
seed = 0
shapes = [
    (16, 64, 16, 16),
    (16, 32, 32, 32),
    (8, 64, 16, 16),
    (8, 32, 32, 32),
    (4, 64, 16, 16),
    (4, 32, 32, 32),
    (2, 64, 16, 16),
    (2, 32, 32, 32),
    (1, 64, 16, 16),
    (1, 32, 32, 32),
]

# ======== 依赖（本地须安装） ========
import zfpy
import zfpy_cuda

# ======== 工具函数 ========
def tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()

def cuda_event_time_s(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0

def wall_time_s(fn):
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return t1 - t0

def median(xs):
    return stats.median(xs) if xs else float("nan")

def safe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return int(getattr(obj, "nbytes", 0))

# ======== 主流程 ========
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

rows = []

for shape in shapes:
    # 原始数据（GPU）
    x = torch.randn(*shape, device=device, dtype=dtype)
    raw_bytes = tensor_bytes(x)

    for tol in tolerances:
        # 计算 GPU 压缩率参数
        rate = zfpy_cuda.compute_rate_from_tolerance(x, tolerance=tol)

        # ---- 预热（不计时）----
        for _ in range(warmup):
            _ = zfpy_cuda.compress_tensor(x, rate)
            comp_tmp = zfpy_cuda.compress_tensor(x, rate)
            _ = zfpy_cuda.decompress_to_tensor(comp_tmp)
            x_cpu_np = x.detach().cpu().numpy()
            comp_np = zfpy.compress_numpy(x_cpu_np, tolerance=tol)
            _ = zfpy.decompress_numpy(comp_np)

        # ---- CUDA 压缩（GPU→CPU bytes）----
        comp_event_times, comp_wall_times, comp_sizes = [], [], []
        for _ in range(repeats):
            # 事件时间（近似核时间；是否包含 host 复制取决于实现）
            t_event = cuda_event_time_s(lambda: zfpy_cuda.compress_tensor(x, rate))
            comp_event_times.append(t_event)

            # 端到端时间（含 GPU→CPU 拷贝 + Python 开销）
            t_wall = wall_time_s(lambda: zfpy_cuda.compress_tensor(x, rate))
            comp_wall_times.append(t_wall)

            # 实际拿一次 bytes 统计大小
            comp_bytes_gpu = zfpy_cuda.compress_tensor(x, rate)
            comp_sizes.append(safe_len(comp_bytes_gpu))

        comp_gpu_event_s = median(comp_event_times)
        comp_e2e_gpu2cpu_s = median(comp_wall_times)
        comp_bytes = int(median(comp_sizes))
        comp_ratio_gpu = (raw_bytes / comp_bytes) if comp_bytes else math.nan

        # 固定一份 CUDA 压缩结果用于解压测试
        comp_ref_gpu = zfpy_cuda.compress_tensor(x, rate)

        # ---- CUDA 解压（解到 GPU）----
        decomp_gpu_times = []
        for _ in range(repeats):
            decomp_gpu_times.append(
                cuda_event_time_s(lambda: zfpy_cuda.decompress_to_tensor(comp_ref_gpu))
            )
        decomp_gpu_s = median(decomp_gpu_times)

        # ---- CUDA 解压 + 搬到 CPU（端到端）----
        decomp_gpu_then_cpu_wall = []
        for _ in range(repeats):
            def _run():
                y_gpu = zfpy_cuda.decompress_to_tensor(comp_ref_gpu)
                _ = y_gpu.cpu()  # GPU→CPU
            decomp_gpu_then_cpu_wall.append(wall_time_s(_run))
        decomp_gpu_then_cpu_s = median(decomp_gpu_then_cpu_wall)

        # ---- CPU 路线（对照组）----
        x_cpu_np = x.detach().cpu().numpy()

        # CPU 压缩
        cpu_comp_times, cpu_sizes = [], []
        for _ in range(repeats):
            cpu_comp_times.append(
                wall_time_s(lambda: zfpy.compress_numpy(x_cpu_np, tolerance=tol))
            )
            cpu_sizes.append(safe_len(zfpy.compress_numpy(x_cpu_np, tolerance=tol)))
        cpu_comp_s = median(cpu_comp_times)
        cpu_comp_bytes = int(median(cpu_sizes))
        cpu_comp_ratio = (raw_bytes / cpu_comp_bytes) if cpu_comp_bytes else math.nan

        # CPU 解压（固定一个 CPU 压缩流）
        comp_np_fixed = zfpy.compress_numpy(x_cpu_np, tolerance=tol)
        cpu_decomp_times = []
        for _ in range(repeats):
            cpu_decomp_times.append(
                wall_time_s(lambda: zfpy.decompress_numpy(comp_np_fixed))
            )
        cpu_decomp_s = median(cpu_decomp_times)

        # ---- 误差评估（与原 x 比；不计时）----
        # CUDA 路线误差（在 CPU 上比较）
        y_gpu = zfpy_cuda.decompress_to_tensor(comp_ref_gpu)
        y_cpu_from_gpu = y_gpu.cpu().numpy()
        if y_cpu_from_gpu.dtype != np.float32:
            y_cpu_from_gpu = y_cpu_from_gpu.astype(np.float32, copy=False)
        diff_cuda = np.abs(y_cpu_from_gpu - x_cpu_np)
        err_max_cuda = float(diff_cuda.max())
        err_mean_cuda = float(diff_cuda.mean())

        # CPU 路线误差（在 CPU 上比较）
        y_cpu = zfpy.decompress_numpy(comp_np_fixed)
        if y_cpu.dtype != np.float32:
            y_cpu = y_cpu.astype(np.float32, copy=False)
        diff_cpu = np.abs(y_cpu - x_cpu_np)
        err_max_cpu = float(diff_cpu.max())
        err_mean_cpu = float(diff_cpu.mean())

        rows.append(dict(
            shape=str(shape),
            numel=int(np.prod(shape)),
            raw_bytes=raw_bytes,
            tolerance=tol,

            # CUDA
            cuda_comp_event_s=comp_gpu_event_s,
            cuda_comp_e2e_gpu2cpu_s=comp_e2e_gpu2cpu_s,
            cuda_decomp_gpu_s=decomp_gpu_s,
            cuda_decomp_gpu_then_cpu_s=decomp_gpu_then_cpu_s,
            cuda_comp_bytes=comp_bytes,
            cuda_comp_ratio=comp_ratio_gpu,
            cuda_err_max=err_max_cuda,
            cuda_err_mean=err_mean_cuda,

            # CPU
            cpu_comp_s=cpu_comp_s,
            cpu_decomp_s=cpu_decomp_s,
            cpu_comp_bytes=cpu_comp_bytes,
            cpu_comp_ratio=cpu_comp_ratio,
            cpu_err_max=err_max_cpu,
            cpu_err_mean=err_mean_cpu,
        ))

# ======== 汇总 / 导出 ========
import pandas as pd
df = pd.DataFrame(rows)
df = df.sort_values(["tolerance", "raw_bytes"]).reset_index(drop=True)
print("\n=== Summary ===")
print(df.to_string(index=False))
csv_path = "zfpy_bench_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV -> {csv_path}")

# ======== 画图（折线图）========
# x 轴用 raw_bytes（更直观反映规模）
import matplotlib.pyplot as plt

def plot_lines(df_sub, x, ys, title, ylabel, outfile):
    plt.figure()
    for y in ys:
        plt.plot(df_sub[x].values, df_sub[y].values, marker="o", label=y)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved plot -> {outfile}")

# 按 tolerance 分面画
for tol in tolerances:
    d = df[df["tolerance"] == tol].copy()

    # 压缩耗时（CUDA 事件 vs 端到端）+ CPU 压缩
    plot_lines(
        d, "raw_bytes",
        ["cuda_comp_event_s", "cuda_comp_e2e_gpu2cpu_s", "cpu_comp_s"],
        f"Compression Time vs Size (tol={tol})",
        "seconds",
        f"plot_compress_tol_{tol}.png"
    )

    # 解压耗时（GPU-only vs GPU->CPU）+ CPU 解压
    plot_lines(
        d, "raw_bytes",
        ["cuda_decomp_gpu_s", "cuda_decomp_gpu_then_cpu_s", "cpu_decomp_s"],
        f"Decompression Time vs Size (tol={tol})",
        "seconds",
        f"plot_decompress_tol_{tol}.png"
    )

    # 压缩率（越大越好）
    plot_lines(
        d, "raw_bytes",
        ["cuda_comp_ratio", "cpu_comp_ratio"],
        f"Compression Ratio vs Size (tol={tol})",
        "ratio (raw_bytes / compressed_bytes)",
        f"plot_ratio_tol_{tol}.png"
    )

# 也给一张按形状标注的图（以 tol 的第一项为例）
tol0 = tolerances[0]
d0 = df[df["tolerance"] == tol0].copy()
plt.figure()
plt.plot(d0["raw_bytes"].values, d0["cuda_comp_e2e_gpu2cpu_s"].values, marker="o", label="cuda_comp_e2e_gpu2cpu_s")
for i, row in d0.iterrows():
    plt.annotate(row["shape"], (row["raw_bytes"], row["cuda_comp_e2e_gpu2cpu_s"]), xytext=(5,5), textcoords="offset points", fontsize=8)
plt.xlabel("raw_bytes")
plt.ylabel("seconds")
plt.title(f"CUDA Compress End-to-End (tol={tol0})")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"plot_cuda_e2e_annot_tol_{tol0}.png", dpi=150)
print(f"Saved plot -> plot_cuda_e2e_annot_tol_{tol0}.png")

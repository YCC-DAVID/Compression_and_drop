import argparse
import time
import statistics as stats
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os

import numpy as np
import torch
import zfpy
from pympler import asizeof


def build_cmp_tensor(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    n_aug: int,
    selected_channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Synthetically build cmp_data analogous to modify_model6.py:
      - Start with output of shape (B * (1 + N_aug), C, H, W)
      - Split into aug (first B*N_aug) and ori (last B)
      - Select first K channels from each augmentation
      - Reshape (B, N_aug, K, H, W) -> (B, N_aug*K, H, W)
      - Concatenate with original along channel dim -> (B, C + N_aug*K, H, W)
    """
    assert selected_channels <= channels, "selected_channels must be <= channels"

    total_batches = batch_size * (1 + n_aug)
    output = torch.randn((total_batches, channels, height, width), dtype=dtype, device=device)

    aug_tensor = output[: batch_size * n_aug]
    ori_tensor = output[batch_size * n_aug :]

    # (B*N_aug, C, H, W) -> (B, N_aug, C, H, W)
    aug_tensor = aug_tensor.view(batch_size, n_aug, channels, height, width)
    # select K channels from each augmentation
    reserved_aug_data = aug_tensor[:, :, :selected_channels, :, :]  # (B, N_aug, K, H, W)
    # flatten augmentations into channel dim: (B, N_aug*K, H, W)
    reserved_aug_flat = reserved_aug_data.reshape(batch_size, n_aug * selected_channels, height, width)
    # concat with original along channel dim
    cmp_data = torch.cat((ori_tensor, reserved_aug_flat), dim=1)
    return cmp_data


def time_step(fn):
    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    return end - start, result


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def compress_worker(np_arr, tol):
    return zfpy.compress_numpy(np_arr, tolerance=float(tol))


def size_worker(blob):
    return asizeof.asizeof(blob)


def main():
    parser = argparse.ArgumentParser(description="Synthetic benchmark for compression pipeline bottlenecks")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n-aug", type=int, default=4, dest="n_aug")
    parser.add_argument("--select-channels", type=int, default=8, dest="select_channels")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modes",
        type=str,
        default="single,chunked,threaded,parallel",
        help="Comma-separated: single, chunked, threaded, parallel",
    )
    parser.add_argument("--chunk-size", type=int, default=16, dest="chunk_size")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--parallelize-asizeof", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    cmp_data = build_cmp_tensor(
        batch_size=args.batch,
        channels=args.channels,
        height=args.height,
        width=args.width,
        n_aug=args.n_aug,
        selected_channels=args.select_channels,
        dtype=dtype,
        device=device,
    )

    intersize_bytes = cmp_data.element_size() * cmp_data.nelement()
    intersize_mb = intersize_bytes / (1024 ** 2)

    # Warmup on full tensor
    for _ in range(args.warmup):
        synchronize_if_cuda(device)
        _np = cmp_data.detach().cpu().numpy()
        _comp = zfpy.compress_numpy(_np, tolerance=args.tolerance)
        _ = asizeof.asizeof(_comp)

    def summarize(name, values):
        total = sum(values)
        mean_v = total / max(1, len(values))
        median_v = stats.median(values) if values else 0.0
        return name, mean_v, median_v, total

    def print_header():
        print("\nSynthetic compression bottleneck benchmark")
        print("-" * 60)
        print(f"Device: {device.type}, dtype: {args.dtype}")
        print(f"Tensor shape: (B={args.batch}, C={args.channels}+{args.n_aug}*{args.select_channels}, H={args.height}, W={args.width})")
        print(f"Uncompressed tensor size: {intersize_mb:.2f} MB")
        print(f"Tolerance: {args.tolerance}")
        print(f"Repeats: {args.repeats} (warmup: {args.warmup})")
        print(f"Chunk size: {args.chunk_size}, Workers: {args.workers}")
        print("-" * 60)

    def print_block(title, s1, s2, s3):
        grand_total = s1[3] + s2[3] + s3[3]
        def line(label, mean_v, median_v, total_v):
            pct = (total_v / grand_total * 100.0) if grand_total > 0 else 0.0
            print(f"{label:>10}: mean {mean_v*1000:.2f} ms | median {median_v*1000:.2f} ms | share {pct:5.1f}%")
        print(f"[{title}]")
        line(*s1)
        line(*s2)
        line(*s3)
        print(f"Total (all steps): {grand_total*1000:.2f} ms")
        print("-" * 60)

    # Helper: split tensor into chunks along batch dim
    def split_tensor_chunks(tensor: torch.Tensor, chunk_size: int):
        num = tensor.shape[0]
        for start in range(0, num, chunk_size):
            end = min(start + chunk_size, num)
            yield tensor[start:end]

    # Mode: single (baseline)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print_header()

    if "single" in modes:
        to_numpy_times = []
        compress_times = []
        asizeof_times = []
        for _ in range(args.repeats):
            synchronize_if_cuda(device)
            t, cmp_np = time_step(lambda: cmp_data.detach().cpu().numpy())
            to_numpy_times.append(t)
            t, comp = time_step(lambda: zfpy.compress_numpy(cmp_np, tolerance=float(args.tolerance)))
            compress_times.append(t)
            t, _ = time_step(lambda: asizeof.asizeof(comp))
            asizeof_times.append(t)
        s1 = summarize("to_numpy", to_numpy_times)
        s2 = summarize("compress", compress_times)
        s3 = summarize("asizeof", asizeof_times)
        print_block("single", s1, s2, s3)

    # Mode: chunked (sequential per chunk)
    if "chunked" in modes:
        to_numpy_times = []
        compress_times = []
        asizeof_times = []
        for _ in range(args.repeats):
            to_numpy_sum = 0.0
            compress_sum = 0.0
            asizeof_sum = 0.0
            for chunk in split_tensor_chunks(cmp_data, args.chunk_size):
                synchronize_if_cuda(device)
                dt, np_chunk = time_step(lambda: chunk.detach().cpu().numpy())
                to_numpy_sum += dt
                dt, comp = time_step(lambda: zfpy.compress_numpy(np_chunk, tolerance=float(args.tolerance)))
                compress_sum += dt
                dt, _ = time_step(lambda: asizeof.asizeof(comp))
                asizeof_sum += dt
            to_numpy_times.append(to_numpy_sum)
            compress_times.append(compress_sum)
            asizeof_times.append(asizeof_sum)
        s1 = summarize("to_numpy", to_numpy_times)
        s2 = summarize("compress", compress_times)
        s3 = summarize("asizeof", asizeof_times)
        print_block("chunked", s1, s2, s3)

    # Mode: parallel (compress chunks in parallel using processes)
    if "parallel" in modes:
        to_numpy_times = []
        compress_times = []  # wall time for parallel compress
        asizeof_times = []

        for _ in range(args.repeats):
            # 1) Copy chunks to CPU numpy (sequential GPU->CPU copies)
            np_chunks = []
            to_numpy_sum = 0.0
            for chunk in split_tensor_chunks(cmp_data, args.chunk_size):
                synchronize_if_cuda(device)
                dt, np_chunk = time_step(lambda: chunk.detach().cpu().numpy())
                to_numpy_sum += dt
                np_chunks.append(np_chunk)
            to_numpy_times.append(to_numpy_sum)

            # 2) Compress in parallel
            start = time.perf_counter()
            compressed_chunks = []
            if args.workers <= 1:
                for arr in np_chunks:
                    compressed_chunks.append(compress_worker(arr, args.tolerance))
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futures = [ex.submit(compress_worker, arr, args.tolerance) for arr in np_chunks]
                    for f in as_completed(futures):
                        compressed_chunks.append(f.result())
            end = time.perf_counter()
            compress_times.append(end - start)

            # 3) Optionally parallelize asizeof (off by default)
            if args.parallelize_asizeof and args.workers > 1:
                start_sz = time.perf_counter()
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    list(ex.map(size_worker, compressed_chunks))
                end_sz = time.perf_counter()
                asizeof_times.append(end_sz - start_sz)
            else:
                start_sz = time.perf_counter()
                for blob in compressed_chunks:
                    _ = asizeof.asizeof(blob)
                end_sz = time.perf_counter()
                asizeof_times.append(end_sz - start_sz)

        s1 = summarize("to_numpy", to_numpy_times)
        s2 = summarize("compress", compress_times)
        s3 = summarize("asizeof", asizeof_times)
        print_block("parallel", s1, s2, s3)

    # Mode: threaded (compress chunks in parallel using threads)
    if "threaded" in modes:
        to_numpy_times = []
        compress_times = []  # wall time for threaded compress
        asizeof_times = []

        for _ in range(args.repeats):
            # 1) Copy chunks to CPU numpy (sequential GPU->CPU copies)
            np_chunks = []
            to_numpy_sum = 0.0
            for chunk in split_tensor_chunks(cmp_data, args.chunk_size):
                synchronize_if_cuda(device)
                dt, np_chunk = time_step(lambda: chunk.detach().cpu().numpy())
                to_numpy_sum += dt
                np_chunks.append(np_chunk)
            to_numpy_times.append(to_numpy_sum)

            # 2) Compress in threads (no pickling overhead)
            start = time.perf_counter()
            if args.workers <= 1:
                compressed_chunks = [compress_worker(arr, args.tolerance) for arr in np_chunks]
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    compressed_chunks = list(ex.map(lambda a: compress_worker(a, args.tolerance), np_chunks))
            end = time.perf_counter()
            compress_times.append(end - start)

            # 3) Optionally threaded asizeof
            if args.parallelize_asizeof and args.workers > 1:
                start_sz = time.perf_counter()
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    list(ex.map(size_worker, compressed_chunks))
                end_sz = time.perf_counter()
                asizeof_times.append(end_sz - start_sz)
            else:
                start_sz = time.perf_counter()
                for blob in compressed_chunks:
                    _ = asizeof.asizeof(blob)
                end_sz = time.perf_counter()
                asizeof_times.append(end_sz - start_sz)

        s1 = summarize("to_numpy", to_numpy_times)
        s2 = summarize("compress", compress_times)
        s3 = summarize("asizeof", asizeof_times)
        print_block("threaded", s1, s2, s3)


if __name__ == "__main__":
    main()


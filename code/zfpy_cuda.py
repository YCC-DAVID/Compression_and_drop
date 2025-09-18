# wrappers.py
# zfp_cuda_wrap.py
# -*- coding: utf-8 -*-
"""
轻量 ZFP-CUDA 包装：
- 自动识别 3D/4D 形状（4D 以 B*C 合并为 3D 走 CUDA；解压后再 view 回 4D）
- 压缩结果带二进制 header（解压只需这段 header，即可自动还原）
- 提供容差→码率的估计函数（基于 zfpy 的 CPU 压缩）
依赖：
- torch
- zfp_cuda  （你的 C++/CUDA 扩展，需导出 compress_array_3d_by_tolerance / decompress_to_torch_tensor）
- zfpy      （仅在 compute_rate_from_tolerance 时需要；多批共用一次即可）
"""

from __future__ import annotations
import struct
from typing import Iterable, Tuple
import torch

# === 外部 CUDA 扩展 ===
import zfp_cuda  # 确保已按你的代码编译安装

# === Header 常量定义 ===
MAGIC = b"Z4PC"      # magic
VERS  = 1            # 版本
DT_FLOAT32 = 0       # 目前只支持 float32
ORDER_NCHW = 0       # NCHW / C-order
MAP_DIRECT3D = 0     # 3D 直接存
MAP_NC_TO_X  = 1     # 4D 的 (N,C,H,W) 映射到 (NX=B*C, NY=H, NZ=W)

# 头部格式：<4sBBBBBf + dims(nd * uint64)
# 含义： magic, vers, dtype, order, ndims, mapmode, rate_f32
_HDR_FMT = "<4sBBBBBf"
_HDR_STATIC_LEN = struct.calcsize(_HDR_FMT)

class ZFPCUDAError(RuntimeError):
    pass

# -------------------------
# 工具：打包 / 解包 header
# -------------------------
def _pack_header(shape: Tuple[int, ...], rate_bpv: float, *,
                 dtype_code: int = DT_FLOAT32,
                 order: int = ORDER_NCHW,
                 mapmode: int = MAP_DIRECT3D) -> bytes:
    nd = len(shape)
    if not (1 <= nd <= 8):
        raise ZFPCUDAError(f"ndims out of range: {nd}")
    head = struct.pack(_HDR_FMT, MAGIC, VERS, dtype_code, order, nd, mapmode, float(rate_bpv))
    dims = struct.pack("<" + "Q"*nd, *[int(x) for x in shape])
    return head + dims

def _unpack_header(buf: bytes, offset: int = 0):
    if len(buf) < offset + _HDR_STATIC_LEN:
        raise ZFPCUDAError("invalid blob: too short for header")
    magic, ver, dtype_code, order, nd, mapmode, rate = struct.unpack_from(_HDR_FMT, buf, offset)
    if magic != MAGIC:
        raise ZFPCUDAError("invalid blob: bad magic")
    if ver != VERS:
        raise ZFPCUDAError(f"unsupported header version: {ver}")
    pos = offset + _HDR_STATIC_LEN
    need = struct.calcsize("<" + "Q"*nd)
    if len(buf) < pos + need:
        raise ZFPCUDAError("invalid blob: truncated dims")
    dims = struct.unpack_from("<" + "Q"*nd, buf, pos)
    pos += need
    return (dims, rate, dtype_code, order, mapmode, pos)

# -------------------------
# 主功能：压缩 / 解压
# -------------------------
def compress_tensor(x: torch.Tensor, rate_bits_per_value: float) -> bytes:
    """
    压缩一个 CUDA float32 张量。支持形状：
      - 4D: [B, C, H, W]  ->  映射为 3D: [B*C, H, W]
      - 3D: [X, Y, Z]     ->  直接 3D
    返回： header + 纯 ZFP 比特流
    """
    if not isinstance(x, torch.Tensor):
        raise ZFPCUDAError("expected a torch.Tensor")
    if not x.is_cuda:
        raise ZFPCUDAError("expected a CUDA tensor")
    if x.dtype != torch.float32:
        raise ZFPCUDAError("expected dtype float32")
    if x.dim() not in (3, 4):
        raise ZFPCUDAError("only 3D or 4D tensors are supported")
    x = x.contiguous()

    if x.dim() == 4:
        B, C, H, W = [int(v) for v in x.shape]
        NX, NY, NZ = B*C, H, W
        mapmode = MAP_NC_TO_X
        orig_shape = (B, C, H, W)
    else:  # 3D
        NX, NY, NZ = [int(v) for v in x.shape]
        mapmode = MAP_DIRECT3D
        orig_shape = (NX, NY, NZ)

    # 调用 CUDA 固定码率压缩（注意你的 C++ 侧把 tolerance 参数当作 rate 用）
    payload = zfp_cuda.compress_array_3d_by_tolerance(x.data_ptr(), NX, NY, NZ, rate_bits_per_value)
    header = _pack_header(orig_shape, rate_bits_per_value, dtype_code=DT_FLOAT32,
                          order=ORDER_NCHW, mapmode=mapmode)
    return header + payload

def decompress_to_tensor(blob: bytes) -> torch.Tensor:
    """
    从 header+payload 的 blob 解压为 CUDA float32 torch.Tensor。
    自动根据 header 还原形状（3D 直接，4D 做 view）。
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ZFPCUDAError("expected bytes-like object")
    dims, rate, dtype_code, order, mapmode, pos = _unpack_header(blob)
    if dtype_code != DT_FLOAT32:
        raise ZFPCUDAError("only float32 supported in this wrapper")
    if order != ORDER_NCHW:
        raise ZFPCUDAError("only NCHW/C-order supported in this wrapper")

    if len(blob) <= pos:
        raise ZFPCUDAError("invalid blob: no payload")
    payload = memoryview(blob)[pos:].tobytes()

    # 解析映射 & 目标 3D 尺寸
    if mapmode == MAP_NC_TO_X:
        if len(dims) != 4:
            raise ZFPCUDAError("bad dims for MAP_NC_TO_X (need 4D)")
        B, C, H, W = [int(d) for d in dims]
        NX, NY, NZ = B*C, H, W
        y3 = zfp_cuda.decompress_to_torch_tensor(payload, NX, NY, NZ, float(rate))
        return y3.view(B, C, H, W)  # 零拷贝视图
    elif mapmode == MAP_DIRECT3D:
        if len(dims) != 3:
            raise ZFPCUDAError("bad dims for MAP_DIRECT3D (need 3D)")
        NX, NY, NZ = [int(d) for d in dims]
        return zfp_cuda.decompress_to_torch_tensor(payload, NX, NY, NZ, float(rate))
    else:
        raise ZFPCUDAError(f"unknown mapmode: {mapmode}")

# -------------------------
# 独立：容差 → 码率（一次算好，多批复用）
# -------------------------
def compute_rate_from_tolerance(sample, tolerance: float) -> float:
    """
    基于 CPU 的 zfpy，在一个“代表性样本”上用容差模式压缩一次，
    以估计固定码率（bits per value）。之后可在 CUDA 路径用该 rate 多批压缩。
    - sample: 支持 numpy.float32 数组，或 torch.Tensor（自动搬到 CPU 并转 numpy）
    - tolerance: 绝对误差阈值（zfpy 的 tolerance 参数）
    返回：估计的平均 bppv（float）
    注意：
      - 为减小 zfpy 容器头开销影响，建议 sample 至少有几万至几十万元素；
      - 结果会依赖数据分布，仅作为“工程上足够好”的近似。
    """
    try:
        import numpy as np
        import zfpy
    except Exception as e:
        raise ZFPCUDAError("compute_rate_from_tolerance requires 'zfpy' and 'numpy' installed") from e

    if hasattr(sample, "detach"):  # torch.Tensor
        t: torch.Tensor = sample  # type: ignore
        if t.dtype != torch.float32:
            raise ZFPCUDAError("sample dtype must be float32")
        if t.is_cuda:
            t = t.detach().cpu()
        arr = t.contiguous().numpy()
    else:
        arr = sample
        if not isinstance(arr, np.ndarray):
            raise ZFPCUDAError("sample must be numpy.ndarray or torch.Tensor")
        if arr.dtype != np.float32:
            raise ZFPCUDAError("sample dtype must be float32")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

    comp = zfpy.compress_numpy(arr, tolerance=float(tolerance))
    bppv = (len(comp) * 8.0) / arr.size
    return float(bppv)

# -------------------------
# 便捷：批量接口（可选）
# -------------------------
def compress_many(tensors: Iterable[torch.Tensor], rate_bits_per_value: float):
    """对多批张量逐个压缩，返回 bytes 迭代器。每个批次各自带 header，可独立解压。"""
    for x in tensors:
        yield compress_tensor(x, rate_bits_per_value)

def decompress_many(blobs: Iterable[bytes]):
    """逐个解压（与 compress_many 对应）。"""
    for b in blobs:
        yield decompress_to_tensor(b)

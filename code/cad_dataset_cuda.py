import time

import torch
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import zfpy_cuda
from typing import Callable, List, Tuple 
from torch.utils.data._utils.collate import default_collate

@torch.no_grad()
def gpu_flip_and_inject(x_full: torch.Tensor, tensor_channel: int, aug_index, p: float = 0.5, gen: torch.Generator = None):
    """
    x_full: [B, C_total, H, W] (cuda)，通道顺序为 [ori, aug]
    tensor_channel: C_ori
    aug_index: 需要被 aug 替换的 ori 通道索引（len == C_aug）
    返回：y_ori [B, C_ori, H, W]
    """
    assert x_full.ndim == 4,f'Expected 4D tensor, got {x_full.shape}D'
    B, C_total, H, W = x_full.shape
    C_ori = int(tensor_channel)
    assert 0 < C_ori <= C_total
    aug_index = torch.as_tensor(aug_index, device=x_full.device)
    C_aug = C_total - C_ori
    assert C_aug == int(aug_index.numel()), f"C_aug={C_aug} 必须等于 len(aug_index)"

    ori = x_full[:, :C_ori, :, :]              # 视图
    aug = x_full[:, C_ori:, :, :]              # [B,C_aug,H,W]

    if p <= 0:
        return ori
    if p >= 1:
        sel = torch.ones(B, dtype=torch.bool, device=x_full.device)
    else:
        sel = (torch.rand((B,), device=x_full.device, generator=gen) < p)

    y = ori.clone()
    if sel.any():
        idx = sel.nonzero(as_tuple=False).squeeze(1)
        flipped = torch.flip(y[idx], dims=[-1])            # 水平翻转（NCHW → 翻 W）
        flipped[:, aug_index, :, :] = aug[idx]             # 用未翻的 aug 覆盖指定通道
        y[idx] = flipped
    return y

@torch.no_grad()
def gpu_add_gaussian_noise(x: torch.Tensor, std: float, p: float = 1.0, gen: torch.Generator = None):
    """
    x: [B,C,H,W] (cuda)
    std: 标准差（按 x 的数值尺度；若 x∈[0,1]，0.01/0.03 等）
    p:   按 batch 维度应用噪声的概率（整张图要么加要么不加）
    """
    if p < 1.0:
        keep = (torch.rand((x.size(0), 1, 1, 1), device=x.device, generator=gen) < p).to(x.dtype)
    else:
        keep = None
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) * std
    return x + (noise if keep is None else noise * keep)

def _gaussian_kernel2d(ks: int, sigma: float, device, dtype):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kern = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kern = kern / kern.sum()
    return kern  # [ks, ks]

@torch.no_grad()
def gpu_gaussian_blur(x: torch.Tensor, kernel_size=(3,3), sigma=(0.1,1.0), gen: torch.Generator = None):
    """
    对每个样本独立随机 sigma 做高斯模糊（与 torchvision.transforms.GaussianBlur 行为近似）
    x: [B,C,H,W] (cuda)
    kernel_size: (kh, kw)（必须奇数）
    sigma: (low, high) 或 float
    """
    B, C, H, W = x.shape
    kh, kw = kernel_size
    assert kh % 2 == 1 and kw % 2 == 1, "kernel_size 必须是奇数"
    out = torch.empty_like(x)

    # 每个样本一个 sigma（与 torchvision 一致）
    if isinstance(sigma, (tuple, list)):
        lo, hi = float(sigma[0]), float(sigma[1])
        sigmas = torch.empty(B, device=x.device)
        sigmas.uniform_(lo, hi, generator=gen)
    else:
        sigmas = torch.full((B,), float(sigma), device=x.device)

    for i in range(B):
        k2d = _gaussian_kernel2d(kh, float(sigmas[i].item()), x.device, x.dtype)   # [kh,kw]
        weight = k2d.view(1,1,kh,kw).repeat(C,1,1,1)                                # [C,1,kh,kw]
        out[i:i+1] = F.conv2d(x[i:i+1], weight, stride=1, padding=(kh//2, kw//2), groups=C)
    return out

@torch.no_grad()
def gpu_random_crop(x: torch.Tensor, size: int, padding: int = 0, pad_value: float = 0.0,
                    gen: torch.Generator = None):
    """
    与 transforms.RandomCrop(size, padding) 类似
    x: [B,C,H,W] (cuda)
    size: 输出边长（正方形）
    padding: 四周常数填充像素
    """
    B, C, H, W = x.shape
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=pad_value)
    Hp, Wp = x.shape[-2:]
    max_y = Hp - size
    max_x = Wp - size
    ys = torch.randint(0, max_y + 1, (B,), device=x.device, generator=gen)
    xs = torch.randint(0, max_x + 1, (B,), device=x.device, generator=gen)
    out = torch.empty((B, C, size, size), device=x.device, dtype=x.dtype)
    # 简洁起见用循环（B 一般不大）；如需完全向量化可用 unfold 实现
    for i in range(B):
        out[i] = x[i, :, ys[i]:ys[i]+size, xs[i]:xs[i]+size]
    return out

def compose_gpu(ops: List[Callable]) -> Callable:
    def _op(x: torch.Tensor, gen: torch.Generator = None):
        for f in ops:
            x = f(x, gen=gen) if f.__code__.co_argcount >= 2 else f(x)
        return x
    return _op


class PayloadDataset(Dataset):
    """
    最简单的 CUDA 线路用 Dataset：仅返回 (payload, label)。
    payload: 压缩后的 bytes / np.uint8 数组等（worker 不做任何解压/增强）
    """
    def __init__(self, payloads, labels):
        assert len(payloads) == len(labels)
        self.payloads = payloads
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.payloads[idx], self.labels[idx]


def collate_payload(batch):
    """
    保持 payload 为 list，labels 堆成 LongTensor
    """
    payloads, labels = zip(*batch)
    labels = default_collate(labels).long()
    # print(labels.shape)
    return list(payloads), labels

@torch.no_grad()
def _decompress_cat_bchw(payloads):
    xs, inner_bs = [], []
    for p in payloads:
        t = zfpy_cuda.decompress_to_tensor(p)    # [C,H,W] 或 [b_i,C,H,W]
        if t.dim() == 3:
            t = t.unsqueeze(0)                   # -> [1,C,H,W]
        elif t.dim() != 4:
            raise RuntimeError(f"expect 3D/4D, got {t.shape}")
        inner_bs.append(t.size(0))
        xs.append(t)
    x = torch.cat(xs, dim=0)                     # -> [∑b_i, C, H, W]
    return x, inner_bs

def _flatten_labels_ragged(labels_list, device):
    # labels_list: List[Tensor或list]，每个长度 = b_i
    parts = [torch.as_tensor(l, device=device).long().view(-1) for l in labels_list]
    return torch.cat(parts, dim=0) 


class ZfpyCudaDecompPrefetch:
    """
    解压(主进程独立 CUDA stream) → 拆分(ori/aug) → 可选 flip+inject → 可选额外 GPU 增强(仅 ori)
    返回 (ori_cuda, y_cuda, dcmp_time_cpu_ms)
    """
    def __init__(self, loader, device="cuda", time_unit="ms",
                 tensor_channel: int = None, aug_index=None, flip_p: float = 0.0,
                 extra_gpu_transform=None, gen: torch.Generator = None):
        assert zfpy_cuda is not None, "需要可用的 zfpy_cuda"
        self.loader = loader
        self.dev = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.dev)
        self.scale = 1.0 if time_unit == "s" else 1000.0
        self.tensor_channel = tensor_channel
        self.aug_index = aug_index
        self.flip_p = float(flip_p)
        self.extra_gpu_transform = extra_gpu_transform  # 只作用在 ori 上
        self.gen = gen

    def __len__(self): return len(self.loader)

                                          # [B,C,H,W] cuda

    def __iter__(self):
        it = iter(self.loader)
        next_pack = None

        def preload():
            nonlocal next_pack
            try:
                payloads, labels = next(it)          # collate: (list[payload], LongTensor[B])
            except StopIteration:
                next_pack = None; return

            # 1) 解压 + y.to(cuda) 计时
            t0 = time.perf_counter()
            with torch.cuda.stream(self.stream):
                x_full, inner_bs = _decompress_cat_bchw(payloads)        # [∑b_i, C, H, W] (cuda)
                y = _flatten_labels_ragged(labels, device=self.dev)              # [B]
            self.stream.synchronize()
            d_ms = (time.perf_counter() - t0) * self.scale
            dcmp_time = torch.tensor([d_ms], dtype=torch.float32)       # CPU 张量（毫秒）

            # 2) 在同一 stream 上：拆分 & flip+inject & 额外增强（只作用 ori）
            with torch.cuda.stream(self.stream):
                if self.tensor_channel is not None:
                    if self.aug_index is not None and self.flip_p > 0.0:
                        ori = gpu_flip_and_inject(x_full, self.tensor_channel, self.aug_index,
                                                   p=self.flip_p, gen=self.gen)   # [B,C_ori,H,W]
                    else:
                        ori = x_full[:, :int(self.tensor_channel), :, :]          # 仅 ori
                else:
                    ori = x_full  # 没有 aug，就把全部当成 ori

                if self.extra_gpu_transform is not None:
                    ori = self.extra_gpu_transform(ori, gen=self.gen)            # 仅对 ori 做其它增强

            next_pack = (ori, y, dcmp_time)  # ori 已是 CUDA 张量

        preload()
        while next_pack is not None:
            torch.cuda.current_stream(self.dev).wait_stream(self.stream)
            pack = next_pack
            preload()
            yield pack
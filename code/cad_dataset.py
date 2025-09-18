import os
from matplotlib.style import available
import torch
import zfpy
try:
    import zfpy_cuda
    ZFPY_CUDA_AVAILABLE = True
except Exception:
    ZFPY_CUDA_AVAILABLE = False
import time
import random
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from PIL import Image
# import statsmodels.api as sm
from typing import List, Optional, Tuple, Union
# from monai.networks.nets import BasicUNet
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets, models

import multiprocessing as mp
from collections import defaultdict

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from scipy.stats import shapiro, kstest, normaltest
from time_logger import TimingContext
import matplotlib.pyplot as plt

imgdir='/scratch/cy65664/workDir/transfer_exp/visual_infor'



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------
# Add the traditional data augmentation
# --------------------------------------------
class CmpDataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, transform=None):
        self.images_tensor = images_tensor
        self.labels_tensor = labels_tensor
        self.transform = transform

    def __len__(self):
        return len(self.images_tensor)

    def __getitem__(self, idx):
        # batch_idx, sample_idx = divmod(idx, 64)
        decompressed_data = zfpy.decompress_numpy(self.images_tensor[idx])
        image = torch.tensor(decompressed_data).squeeze(0)  # 转换为 PyTorch 张量
        # image = torch.from_numpy(decompressed_data)

        label = self.labels_tensor[idx]#[batch_idx][sample_idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

class CmpBatchDataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, transform=None):
        self.images_tensor = images_tensor
        self.labels_tensor = labels_tensor
        self.transform = transform

    def __len__(self):
        return len(self.images_tensor)

    def __getitem__(self, idx):
        # with torch.random.fork_rng():  # 保存和恢复随机状态
        #     current_time_seed = int(time.time())
        #     torch.manual_seed(current_time_seed)
        # batch_idx, sample_idx = divmod(idx, 64)
        decompressed_batch = zfpy.decompress_numpy(self.images_tensor[idx])
        ideal_size = round(50000/len(self.images_tensor))
        image = torch.tensor(decompressed_batch)  # 转换为 PyTorch 张量
        label = self.labels_tensor[idx]#[batch_idx][sample_idx]
        if image.shape[0] != ideal_size:
            image = self.pad_tensor(image,ideal_size)
            label = self.pad_tensor(label,ideal_size)
        # indices = torch.randperm(image.size(0))
        # image_shuffled = image[indices]
        # label_shuffled = label[indices]
        # label = self.labels_tensor[idx][ord]#[batch_idx][sample_idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def pad_tensor(self, tensor, target_size):
        """将 Tensor 复制补齐到指定大小"""
        current_size = tensor.shape[0]
        if current_size < target_size:
            # 计算需要复制的次数
            num_repeats = target_size // current_size
            remainder = target_size % current_size

            # 复制 Tensor 并拼接
            tensor = torch.cat([tensor] * num_repeats + [tensor[:remainder]], dim=0)
        
        return tensor
    



class valDataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, transform=None):
        self.images_tensor = images_tensor
        self.labels_tensor = labels_tensor
        self.transform = transform

    def __len__(self):
        return len(self.images_tensor)

    def __getitem__(self, idx):
        image = self.images_tensor[idx].squeeze(0)  # 转换为 PyTorch 张量
        label = self.labels_tensor[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 
class CubicCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, gamma=3.0):
        """
        自定义的Cubic + CosineAnnealing学习率调度器.
        
        参数：
        optimizer: torch.optim.Optimizer
            优化器实例。
        T_max: int
            学习率的最大周期，即从最高学习率衰减到最低学习率的总步数。
        eta_min: float (default: 0)
            最小学习率，学习率衰减到此值后不再降低。
        last_epoch: int (default: -1)
            当前的epoch数目，-1表示从头开始。
        gamma: float (default: 3.0)
            用于立方缩放的因子，默认值为3表示立方缩放。
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.gamma = gamma  # 控制立方缩放强度
        super(CubicCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        获取每个epoch/batch的学习率，结合立方缩放和余弦调度。
        """
        # 当前的epoch/step数
        T_cur = self.last_epoch
        # 余弦学习率调度公式
        cos_inner = np.pi * (T_cur / self.T_max)  # pi * 当前epoch / 最大epoch
        cos_out = np.cos(cos_inner)  # 计算cos( pi * T_cur / T_max )
        
        # 获取基础学习率 (根据当前的cos值进行调整)
        base_lrs = [self.eta_min + (base_lr - self.eta_min) * (1 + cos_out) / 2
                    for base_lr in self.base_lrs]
        
        # 进行立方缩放调整的学习率
        cubic_scaled_lrs = [lr * (1-T_cur / self.T_max) ** self.gamma for lr in base_lrs]
        
        return cubic_scaled_lrs
    

class LayerWiseCosineScheduler:
    def __init__(self, model, optimizer,args, base_lr=0.1, lr_decay=0.9, t_max=100, eta_min_factor=0.1):
        """
        初始化逐层学习率调度器。

        参数：
        - model: nn.Module，待训练的模型。
        - optimizer: torch.optim.Optimizer，优化器。
        - base_lr: float，每层初始学习率的起始值。
        - lr_decay: float，每层学习率递减因子。
        - t_max: int，余弦调度器的最大周期数。
        - eta_min_factor: float，最低学习率是初始学习率的比例。
        """
        self.optimizer = optimizer
        self.schedulers = []
        self._init_schedulers(model, base_lr, lr_decay, t_max, eta_min_factor, args)

    def _init_schedulers(self, model, base_lr, lr_decay, t_max, eta_min_factor,args):
        """初始化每一层的调度器。"""
        layerwise_params = []
        current_lr = base_lr
        for name, param in model.named_parameters():
            if param.requires_grad:
                layerwise_params.append({"params": param, "lr": current_lr})
                current_lr *= lr_decay  # 每层学习率递减

        # 设置优化器的参数组
        self.optimizer.param_groups = layerwise_params

        # 创建逐层调度器
        self.schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=group["lr"] * eta_min_factor)
            for group in self.optimizer.param_groups
        ]

    def step(self):
        """更新所有层的学习率。"""
        for scheduler in self.schedulers:
            scheduler.step()

    def get_layer_lrs(self):
        """获取每层的当前学习率。"""
        return [group["lr"] for group in self.optimizer.param_groups]


class AddGaussianNoise:

    def __init__(self, mean=0.0, std=1.0):
        """
        初始化高斯噪声参数
        - mean: 噪声的均值
        - std: 噪声的标准差
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        在输入张量上添加高斯噪声
        - tensor: 输入的图像张量，形状为 (C, H, W)
        """
        if len(tensor.size()) == 4:
            noise = torch.randn(tensor.size(0), 1, tensor.size(2), tensor.size(3)) * self.std + self.mean
        else:
            noise = torch.randn_like(tensor) * self.std + self.mean

        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# ------------------------------------------------------------
# Add the data augmentation with augementation tensor replace
# ------------------------------------------------------------
class AugmentedDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        参数:
        - data: 输入数据，形状为 (N, C, H, W)
        - transform: 数据增强的变换函数
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        original_sample, label = self.data[idx]  # 解包元组

        # 对样本应用数据增强
        if self.transform:
            augmented_sample = self.transform(original_sample)
        else:
            augmented_sample = transforms.ToTensor(original_sample)
        original_sample = transforms.ToTensor()(original_sample)
        # input = torch.cat((augmented_sample, original_sample), dim=0)

        label_tensor = torch.tensor([label], dtype=torch.long)
        # target = torch.cat((label_tensor, label_tensor), dim=0)

        return augmented_sample, original_sample, label_tensor
    

class EncodableRandomCrop(torch.nn.Module):
    """Randomly crop the image while updating a mask to encode the cropped region."""
    
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop."""
        _, h, w = TF.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.size = tuple(size) if isinstance(size, (list, tuple)) else (size, size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img, mask):
        """
        Args:
            img (Tensor): Image to be cropped.
            mask (Tensor): Mask to be updated for the cropped region.

        Returns:
            Tuple[Tensor, Tensor]: Cropped image and updated mask.
        """
        if isinstance(self.padding, int):
            padding = (self.padding, self.padding, self.padding, self.padding)
        elif isinstance(self.padding, (tuple, list)) and len(self.padding) == 4:
            padding = tuple(self.padding)
        else:
            raise ValueError("padding must be an int or a tuple of four ints.")

        img = F.pad(img, padding, mode=self.padding_mode, value=self.fill, )
        # mask = F.pad(mask, padding, mode=self.padding_mode,value=0)

        _, height, width = TF.get_dimensions(img)

        # Pad if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.padding_mode,self.fill)

        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.padding_mode, self.fill)
          
        # Get crop parameters and apply crop
        i, j, h, w = self.get_params(img, self.size)
        img = TF.crop(img, i, j, h, w)
        # mask.fill_(0)
        new_matrix = torch.zeros_like(mask)

        # 将原矩阵的矩形区域值复制到新矩阵
        new_matrix[i:i + h, j:j + w] = mask[i:i + h, j:j + w]

        # 替换原矩阵
        mask = new_matrix
        # mask[i:i + h, j:j + w] = 1

        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


class EncodableRandomHorizontalFlip(torch.nn.Module):
    """Randomly flip the image horizontally while updating a mask to encode the flip state."""
    
    def __init__(self, p=0.5):
        """
        Args:
            p (float): Probability of the image being flipped.
        """
        super().__init__()
        self.p = p

    def forward(self, img, mask = None, layer_index=None, aug_tensor=None):
        """
        Args:
            img (Tensor): Image to be flipped.
            mask (Tensor): Mask to be updated for the flip state.

        Returns:
            Tuple[Tensor, Tensor]: Flipped image and updated mask.
        """
        if torch.rand(1).item() < self.p:
            img = TF.hflip(img)
            if aug_tensor is not None:
                img[:,layer_index,:,:] = aug_tensor
            # mask = TF.hflip(mask)
            # mask = -mask
        return img , mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
    


class EncodableMultiAngleRotation(torch.nn.Module):
    def __init__(self, angles=(0, 90, 180, 270)):
        super().__init__()
        self.angles = angles

    def forward(self, img, mask=None, layer_index=None, aug_tensor=None):
        """
        If aug_tensor is None:
            return all 4 rotated versions
        Else:
            use one randomly selected angle and inject aug_tensor at layer_index
        """
        if aug_tensor is None:
            # Full rotation mode: generate all 4 rotated versions
            aug_images = []
            aug_masks = [] if mask is not None else None

            for angle in self.angles:
                rotated_img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                aug_images.append(rotated_img)

                if mask is not None:
                    rotated_mask = TF.rotate(mask.unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
                    aug_masks.append(rotated_mask)

            aug_data = torch.stack(aug_images, dim=0)
            aug_mask = torch.stack(aug_masks, dim=0) if mask is not None else None
            return aug_data, aug_mask
        else:
            # In injected mode: pick one random angle and inject aug_tensor into that rotated version
            angle = random.choice(self.angles)
            rotated_img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)

            if layer_index is not None:
                rotated_img[:, layer_index, :, :] = aug_tensor.squeeze(0)  # make sure shape matches

            rotated_mask = None
            if mask is not None:
                rotated_mask = TF.rotate(mask.unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

            return rotated_img, rotated_mask


def apply_flip_rotation(img, flip=False, angle=None):
    if flip:
        img = TF.hflip(img)
    if angle is not None:
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
    return img


class CustomAugmentedDataset(Dataset):
    """
    A dataset that applies encodable random crop and horizontal flip while generating masks.
    """
    def __init__(self, data, crop_size=(32, 32), padding=4, flip_prob=0.5, transform=None):
        """
        Args:
            data (list): List of (image, label) tuples.
            crop_size (tuple): The size of the crop (height, width).
            padding (int): Padding for the crop operation.
            flip_prob (float): Probability of horizontal flip.
            transform (callable, optional): Additional transforms to apply after crop and flip.
        """
        self.data = data
        self.crop = EncodableRandomCrop(size=crop_size, padding=padding)
        self.flip = EncodableRandomHorizontalFlip(p=flip_prob)
        # self.rotation = EncodableMultiAngleRotation(angles=(0, 0))
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def apply_flip_rotation(self, img, flip=False, angle=None):
        if flip:
            img = TF.hflip(img)
        if angle is not None:
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        return img

    def __getitem__(self, idx):
        ori_img, label = self.data[idx]  # (PIL or Tensor, int)

        # Ensure tensor
        if not isinstance(ori_img, torch.Tensor):
            ori_img = transforms.ToTensor()(ori_img)

        # Create mask (for compatibility; all ones)
        _, height, width = ori_img.size()
        mask = torch.ones((height, width), dtype=torch.float32)

        # Build 5 augmentation versions
        # aug_versions = []
        # flip_rotation_combinations = [
        #     (True, None),     # flip only
        #     (False, 90),      # rot90
        #     (False, 270),     # rot270
        #     (True, 90),       # flip + rot90
        #     (True, 270),      # flip + rot270
        # ]

        # for flip_flag, rot_angle in flip_rotation_combinations:
        #     aug_img = self.apply_flip_rotation(ori_img.clone(), flip=flip_flag, angle=rot_angle)
        #     if self.transform:
        #         aug_img = self.transform(aug_img)
        #     aug_versions.append(aug_img)
        aug_img, mask = self.flip(ori_img, mask)

        # aug_imgs = torch.stack(aug_versions, dim=0)  # shape: (5, C, H, W)

        if self.transform:
            ori_img = self.transform(ori_img)
            aug_img = self.transform(aug_img)

        label = torch.tensor(label, dtype=torch.long)

        return aug_img, ori_img, label, mask



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
    


class CustomCmpBatchDataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, ori_tensor_channel=None, aug_index=None, transform=None, crop_size=(32, 32), padding=4, flip_prob=0.5,num_aug_versions=5,dcmp=False):
        self.images_tensor = images_tensor
        self.labels_tensor = labels_tensor
        self.crop = EncodableRandomCrop(size=crop_size, padding=padding)
        self.flip = TimedTransform(EncodableRandomHorizontalFlip(p=flip_prob),name="SelfCustomFlip",cuda_sync=False)
        self.tensor_channel = ori_tensor_channel
        self.aug_index = aug_index
        self.num_aug_versions = num_aug_versions
        self.aug_channel_range = None  # 稍后推断
        self.transform = transform
        self.dcmp_mth = dcmp

    def __len__(self):
        return len(self.images_tensor)
    
    def apply_spatial_aug(self,tensor, aug_code):
        if aug_code in [0, 3, 4]:  # flip
            tensor = TF.hflip(tensor)
        if aug_code in [1, 3]:     # rot90
            tensor = TF.rotate(tensor, 90, interpolation=TF.InterpolationMode.BILINEAR)
        elif aug_code in [2, 4]:   # rot270
            tensor = TF.rotate(tensor, 270, interpolation=TF.InterpolationMode.BILINEAR)
        return tensor
    
    def _decompress_with_timing(self, packed):
        """
        仅负责解压 + 计时，把耗时投递给 decomp_meter。
        - CUDA 路径用 cuda events 计 GPU kernel 时间
        - CPU 路径用 perf_counter
        """
        if ZFPY_CUDA_AVAILABLE and self.dcmp_mth:
            # CUDA 计时：事件更准确；需同步
            start = time.perf_counter()

            unpack_image = zfpy_cuda.decompress_to_tensor(packed)
            unpack_image = unpack_image.to(device="cpu")  # 确保在正确的设备上
            end = time.perf_counter()
            dcmp_t = end - start # 秒
            dcmp_backend = "zfpy_cuda"
            
        else:
            t0 = time.perf_counter()
            decompressed = zfpy.decompress_numpy(packed)
            unpack_image = torch.tensor(decompressed)
            dcmp_t = time.perf_counter() - t0
            dcmp_backend = "zfpy_cpu"
        
        if TimingContext.logger is not None:
            TimingContext.logger.acc_decompress(dcmp_t)
        return unpack_image,dcmp_t

    def __getitem__(self, idx):
        # random.seed(time.time())
        # ord =  random.randint(0, 63)
        # batch_idx, sample_idx = divmod(idx, 64)
        # if zfpy_cuda is available:
        #     unpack_image = zfpy_cuda.decompress_to_tensor(self.images_tensor[idx])
        # else:
        #     decompressed_batch = zfpy.decompress_numpy(self.images_tensor[idx])
        #     unpack_image = torch.tensor(decompressed_batch)#.squeeze(0)  # 转换为 PyTorch 张量
        unpack_image,dcmp_t = self._decompress_with_timing(self.images_tensor[idx])
        if self.aug_index is not None:
            ori_tensor = unpack_image[:,:self.tensor_channel,:,:]
            aug_tensor = unpack_image[:,self.tensor_channel:,:,:]
            # if self.aug_channel_range is None:
            #     total_aug_channels = aug_tensor.shape[1]
            #     assert total_aug_channels % self.num_aug_versions == 0, "通道数不能整除增强数"
            #     self.aug_channel_range = total_aug_channels // self.num_aug_versions
        else:
            ori_tensor = unpack_image
            aug_tensor = None
        # image = unpack_image[:,:self.tensor_channel,:,:]
        # aug_layer = unpack_image[:,self.tensor_channel:,:,:]
        # aug_code = random.randint(0, 4)
        
        # image = self.apply_spatial_aug(ori_tensor.clone(), aug_code)

        # if self.aug_index is not None:
        #     start = aug_code * self.aug_channel_range
        #     end = start + self.aug_channel_range
        #     for i, channel in enumerate(self.aug_index):
        #         image[:, channel, :, :] = aug_tensor[:, start + i, :, :]
        image, mask = self.flip(ori_tensor, layer_index=self.aug_index, aug_tensor=aug_tensor)
        image = image.squeeze(0)
        
        label = self.labels_tensor[idx]#[batch_idx][sample_idx]

        if self.transform:
            image = self.transform(image)

        return image, label, dcmp_t

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
    noise = torch.randn_like(x, generator=gen) * std
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


class CudaCmp_Dataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, ori_tensor_channel=None, aug_index=None, transform=None, crop_size=(32, 32), padding=4, flip_prob=0.5,num_aug_versions=5,dcmp=False):
        self.images_tensor = images_tensor
        self.labels_tensor = labels_tensor
        self.crop = EncodableRandomCrop(size=crop_size, padding=padding)
        self.flip = EncodableRandomHorizontalFlip(p=flip_prob)
        self.tensor_channel = ori_tensor_channel
        self.aug_index = aug_index
        self.num_aug_versions = num_aug_versions
        self.aug_channel_range = None  # 稍后推断
        self.transform = transform
        self.dcmp_mth = dcmp

    def __len__(self):
        return len(self.images_tensor)
    
    # def apply_spatial_aug(self,tensor, aug_code):
    #     if aug_code in [0, 3, 4]:  # flip
    #         tensor = TF.hflip(tensor)
    #     if aug_code in [1, 3]:     # rot90
    #         tensor = TF.rotate(tensor, 90, interpolation=TF.InterpolationMode.BILINEAR)
    #     elif aug_code in [2, 4]:   # rot270
    #         tensor = TF.rotate(tensor, 270, interpolation=TF.InterpolationMode.BILINEAR)
    #     return tensor
    
    def _decompress_with_timing(self, packed):
        """
        仅负责解压 + 计时，把耗时投递给 decomp_meter。
        - CUDA 路径用 cuda events 计 GPU kernel 时间
        - CPU 路径用 perf_counter
        """
        if ZFPY_CUDA_AVAILABLE and self.dcmp_mth:
            # CUDA 计时：事件更准确；需同步
            start = time.perf_counter()

            unpack_image = zfpy_cuda.decompress_to_tensor(packed)
            end = time.perf_counter()

            dcmp_t = (end - start) * 1000.0  # 毫秒
            
        else:
            t0 = time.perf_counter()
            decompressed = zfpy.decompress_numpy(packed)
            unpack_image = torch.tensor(decompressed)
            dcmp_t = time.perf_counter() - t0
        return unpack_image,dcmp_t

    def __getitem__(self, idx):
        # random.seed(time.time())
        # ord =  random.randint(0, 63)
        # batch_idx, sample_idx = divmod(idx, 64)
        # if zfpy_cuda is available:
        #     unpack_image = zfpy_cuda.decompress_to_tensor(self.images_tensor[idx])
        # else:
        #     decompressed_batch = zfpy.decompress_numpy(self.images_tensor[idx])
        #     unpack_image = torch.tensor(decompressed_batch)#.squeeze(0)  # 转换为 PyTorch 张量
        unpack_image,dcmp_t = self._decompress_with_timing(self.images_tensor[idx])
        if self.aug_index is not None:
            ori_tensor = unpack_image[:,:self.tensor_channel,:,:]
            aug_tensor = unpack_image[:,self.tensor_channel:,:,:]
            # if self.aug_channel_range is None:
            #     total_aug_channels = aug_tensor.shape[1]
            #     assert total_aug_channels % self.num_aug_versions == 0, "通道数不能整除增强数"
            #     self.aug_channel_range = total_aug_channels // self.num_aug_versions
        else:
            ori_tensor = unpack_image
            aug_tensor = None
        # image = unpack_image[:,:self.tensor_channel,:,:]
        # aug_layer = unpack_image[:,self.tensor_channel:,:,:]
        # aug_code = random.randint(0, 4)
        
        # image = self.apply_spatial_aug(ori_tensor.clone(), aug_code)

        # if self.aug_index is not None:
        #     start = aug_code * self.aug_channel_range
        #     end = start + self.aug_channel_range
        #     for i, channel in enumerate(self.aug_index):
        #         image[:, channel, :, :] = aug_tensor[:, start + i, :, :]
        image, mask = self.flip(ori_tensor, layer_index=self.aug_index, aug_tensor=aug_tensor)
        image = image.squeeze(0)
        
        label = self.labels_tensor[idx]#[batch_idx][sample_idx]

        if self.transform:
            image = self.transform(image)

        return image, label, dcmp_t
class CudaCmp_Dataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, ori_tensor_channel=None, aug_index=None,
                 transform=None, crop_size=(32, 32), padding=4, flip_prob=0.5,
                 num_aug_versions=5, dcmp=False, return_payload_only=False):
        self.images_tensor = images_tensor        # 压缩后的 payload 列表/数组
        self.labels_tensor = labels_tensor
        self.crop = EncodableRandomCrop(size=crop_size, padding=padding)
        self.flip = EncodableRandomHorizontalFlip(p=flip_prob)
        self.tensor_channel = ori_tensor_channel  # 原通道数
        self.aug_index = aug_index                # 替换的增强通道索引
        self.num_aug_versions = num_aug_versions
        self.transform = transform                # 若是 torchvision 的 CPU 变换，放到 wrapper 里谨慎使用
        self.dcmp_mth = dcmp                      # 原有开关(不再用在Dataset侧)
        self.return_payload_only = return_payload_only

    def __len__(self):
        return len(self.images_tensor)

    # 原来的 _decompress_with_timing 保留，但我们不再在 __getitem__ 里调用

    def __getitem__(self, idx):
        if self.return_payload_only:
            # 关键：只返回压缩块 + label，不在 worker 解压
            return self.images_tensor[idx], self.labels_tensor[idx]
        else:
            # 需要兼容旧路径时才走（不推荐与本方案一起使用）
            unpack_image, dcmp_t = self._decompress_with_timing(self.images_tensor[idx])
            if self.aug_index is not None:
                ori_tensor = unpack_image[:, :self.tensor_channel, :, :]
                aug_tensor = unpack_image[:, self.tensor_channel:, :, :]
            else:
                ori_tensor, aug_tensor = unpack_image, None
            image, mask = self.flip(ori_tensor, layer_index=self.aug_index, aug_tensor=aug_tensor)
            image = image.squeeze(0)
            if self.transform:
                image = self.transform(image)
            label = self.labels_tensor[idx]
            return image, label, dcmp_t


# --------------------------------------------
# Add the data augmentation with learnable method
# --------------------------------------------
class CustomTrainer:

    def __init__(self, model, optimizer,logger=None, scheduler=None, criterion=None, device="cuda"):
        """
        初始化训练所需的对象
        :param model: 要训练的模型
        :param optimizer: 优化器
        :param scheduler: 学习率调度器（可选）
        :param criterion: 损失函数（可选）
        :param device: 设备 (cpu 或 cuda)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.run_batch = 0
        self.val_batch = 0
        self.total_loss = 0.0
        self.batch_loss = 0.0
        self.runtime = time.time()
        self.log = logger
        self.epoch = 0
        self.min_loss = float('inf')
        self.val_loss = 0.0
        self.psnr = 0.0
        self.ssim = 0.0

    def to(self, device):
        """
        将模型和相关设备迁移到指定设备
        """
        self.device = device
        self.model = self.model.to(device)

    def train(self, input_data, label_data,trans_mask = None):
        # batch_loss = 0.0
        # num_batches = ori_data.size(0)
        input_data = input_data.to(device)
        label_data = label_data.to(device)
        # if self.epoch == 20:
        #     ifnormal = self.test_normality(input_data, method='kstest')

        #------------------------------------
        # training data normalization process
        #------------------------------------
        # batch_min,batch_max = self.calculate_batch_minmax(input_data)
        # input_data,_,_ = self.min_max_normalize(input_data, batch_min, batch_max)
        # label_data,_,_ = self.min_max_normalize(label_data, batch_min, batch_max)


        #----------------------------
        # if augmentation mask exists
        #----------------------------
        height = input_data.size(2)  # 或 x.shape[2]
        width = input_data.size(3)
        if trans_mask is not None and trans_mask.numel() > 0:
            trans_mask = trans_mask.unsqueeze(1).to(device)
            trans_mask = F.interpolate(trans_mask,
                                        size=(height, width),
                                        mode='bilinear', 
                                        align_corners=False)
            
        #----------------------------
        # training process
        #----------------------------
        self.model.train()
        pred = self.model(input_data)
        loss = self.criterion(pred, label_data)
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()
        self.total_loss += loss.item()
        current_loss = loss.item()

        if self.run_batch % 50 == 0 and self.scheduler is not None:
            self.scheduler.step()
            batch_loss = self.total_loss / 50
            self.total_loss = 0.0
            current_time = time.time()
            cost_time = current_time - self.runtime
            print(f"Epoch {self.epoch}, \t Batch {self.run_batch}, \t Time: {cost_time:.3f}, \t Loss: {batch_loss:.4f}, \t Current lr: {self.optimizer.param_groups[0]['lr']:.5e}")
            self.runtime = time.time()
        if self.log is not None:
            self.log.log({f'Unet training loss':current_loss}) 
        self.run_batch += 1
                    
        # average_loss = total_loss / num_batches
        

    def validation(self, input_data, label_data,trans_mask = None):
        subdir = f'Cmp4Train_exp/pytorch_resnet_cifar10/'
        save_dir = subdir + 'temp_unet_model'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.batch_loss = 0.0
        # num_batches = input_data.size(0)
        val_data = input_data.to(device)
        label_data = label_data.to(device)

        # ------------------------------------
        # validation data normalization process
        # ------------------------------------
        # batch_min,batch_max = self.calculate_batch_minmax(input_data)
        # val_data,_,_ = self.min_max_normalize(val_data, batch_min, batch_max)
        # label_data,_,_ = self.min_max_normalize(label_data, batch_min, batch_max)
        # label_data = self.normalize(label_data, batch_min, batch_max)

        #----------------------------
        # if augmentation mask exists
        #----------------------------
        height = input_data.size(2)  # 或 x.shape[2]
        width = input_data.size(3)
        if trans_mask is not None and trans_mask.numel() > 0:
            trans_mask = trans_mask.unsqueeze(1).to(device)
            trans_mask = F.interpolate(trans_mask, size=(height, width), mode='bilinear', align_corners=False)
            # val_data = torch.cat((val_data,trans_resized),dim=1)

        # --------------------------
        # validation process
        # --------------------------
        pred = self.model(val_data)
        # pred_denorm = self.denormalize(tensor_mean, tensor_std, pred)
        # pred = self.min_max_denormalize(batch_min, batch_max, pred)
        # print(torch.cuda.memory_allocated()/1024**2)
        loss = self.criterion(pred, label_data)
        val_loss = loss.item()
        self.val_loss += val_loss
  

        #----------------------------
        # calculate PSNR and SSIM
        #----------------------------
        mse = F.mse_loss(pred, label_data, reduction='mean')  # 计算 MSE
        max_val = max(torch.max(label_data),torch.max(pred))
        # psnr = 10 * torch.log10(max_val**2 / torch.sqrt(mse))   # 计算 PSNR
        # psnr = self.calculate_batch_psnr(pred, label_data)
        psnr_val = psnr(pred, label_data, data_range=max_val)
        self.psnr += psnr_val.detach().cpu().numpy()
        ssim_val = ssim(pred, label_data, data_range=max_val, reduction='elementwise_mean')
        self.ssim += ssim_val.detach().cpu().numpy()
        self.val_batch += 1
        # print(torch.cuda.memory_allocated()/1024**2)

        if self.val_batch % 50 == 0 :

            val_loss = self.val_loss / 50
            psnr_val = self.psnr / 50
            ssim_val = self.ssim / 50
            self.val_loss = 0.0
            self.psnr = 0.0
            self.ssim = 0.0
            # current_time = time.time()
            # cost_time = current_time - self.runtime
            # print(f"Epoch {self.epoch}, \t Batch {self.run_batch}, \t Time: {cost_time:.3f}, \t Loss: {val_loss:.4f}, \t Current lr: {self.optimizer.param_groups[0]['lr']:.5e}")
            # self.runtime = time.time()
            print(f"Validation Loss: {val_loss:.4f}, \t PSNR: {psnr_val:.2f} dB, \t SSIM: {ssim_val:.4f}")
        
            if self.log is not None:
                self.log.log({'Unet validate loss':val_loss,'Unet validate PSNR':psnr_val,'SSIM':ssim_val}) #,'Max value':max_val
        

        #----------------------------
        # save the best model
        #----------------------------
        is_best = self.min_loss > val_loss
        self.min_loss = min(val_loss, self.min_loss)

        if is_best:
            model_save=self.model
            self.save_checkpoint({
                'state_dict': model_save.state_dict(),
                'min_loss': self.min_loss,
            }, filename=os.path.join(save_dir, 'linear_model.th'))


    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """
        Save the training model
        """
        torch.save(state, filename)
    
    def calculate_batch_psnr(self, pred, label_data):
        '''
        批量计算 PSNR
        Args:
            pred (Tensor): 生成的张量 (B, C, H, W)
            label_data (Tensor): 参考张量 (B, C, H, W)
            max_val (float): 数据的最大可能值，默认 1.0（标准化到 [0, 1]）

        Returns:
            float: 平均 PSNR 值
        '''
        batch_size = pred.shape[0]
        psnr_values = []

        for i in range(batch_size):
            max_val = max(pred[i].max().item(), label_data[i].max().item())
            mse = F.mse_loss(pred[i], label_data[i], reduction='mean')
            if mse == 0:
                psnr_values.append(float('inf'))
            else:
                psnr = 10 * torch.log10(max_val ** 2 / mse)
                psnr_values.append(psnr.item())

        return sum(psnr_values) / len(psnr_values)

    def test_normality(self, tensor, method='shapiro'):
        """
        Identify if the value in Tensor is normal distribution
        Args:
            tensor (torch.Tensor): 输入 Tensor
            method (str): 正态性检验方法，可选 'shapiro', 'kstest', 'normaltest'
        Returns:
            bool: 是否符合正态分布
            dict: 检验统计量和 p 值
        """
        # 将 Tensor 转换为 NumPy 数组
        data = tensor.cpu().numpy().flatten()
        
        # 选择正态性检验方法
        if method == 'shapiro':
            stat, p_value = shapiro(data)
        elif method == 'kstest':
            # 使用 Kolmogorov-Smirnov 检验
            stat, p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
        elif method == 'normaltest':
            # 使用 D'Agostino 和 Pearson 的正态性检验
            stat, p_value = normaltest(data)
        else:
            raise ValueError("Invalid method. Choose from 'shapiro', 'kstest', or 'normaltest'")
        
        # 判断是否符合正态分布（显著性水平 0.05）
        is_normal = p_value > 0.05
        self.visualize_distribution(tensor)
        
        return is_normal, {"statistic": stat, "p_value": p_value}
    
    def visualize_distribution(self,tensor):
        """
        可视化 Tensor 数据分布和 QQ 图
        Args:
            tensor (torch.Tensor): 输入 Tensor
        """
        data = tensor.cpu().numpy().flatten()
        
        # 绘制直方图
        plt.figure(figsize=(6, 6))
        plt.hist(data, bins=200, density=True, alpha=0.6, color='g')
        plt.title("Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig(imgdir+"tensor_histogram.png")
        
        
        plt.savefig(imgdir+"distribution.png")
    
    def calculate_batch_minmax(self, batch_data):
        """
        计算每个样本的 min 和 max 值
        Args:
            batch_data (torch.Tensor): 输入 batch 数据，形状为 (B, C, H, W)。
        Returns:
            torch.Tensor: 每个样本的最小值，形状为 (B,)。
            torch.Tensor: 每个样本的最大值，形状为 (B,)。
        """
        # 沿 (C, H, W) 维度计算最小值和最大值
        batch_min = torch.amin(batch_data, dim=(0, 2, 3))  # 结果形状为 (B,)
        batch_max = torch.amax(batch_data, dim=(0, 2, 3))  # 结果形状为 (B,)
        return batch_min, batch_max

    def min_max_normalize(self,data, min_val=None, max_val=None, feature_range=(0, 1)):
        """
        对数据进行 Min-Max 归一化，同时支持指定 min/max。
        Args:
            data (torch.Tensor): 输入数据。
            min_val (float, optional): 数据的最小值，若为 None 则从 data 中计算。
            max_val (float, optional): 数据的最大值，若为 None 则从 data 中计算。
            feature_range (tuple): 目标范围，默认为 (0, 1)。
        Returns:
            torch.Tensor: 归一化后的数据。
            float: 使用的最小值。
            float: 使用的最大值。
        """
        if min_val is None or max_val is None:
            min_val, max_val = data.min(), data.max()
        data_normalized = (data - min_val) / (max_val - min_val)
        if feature_range != (0, 1):
            data_normalized = data_normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        return data_normalized, min_val, max_val

    def min_max_denormalize(self, normalized_data, min_val, max_val, feature_range=(0, 1)):
        """
        对 Min-Max 归一化后的数据进行反归一化。
        Args:
            normalized_data (torch.Tensor): 已归一化的数据。
            min_val (float): 数据的最小值。
            max_val (float): 数据的最大值。
            feature_range (tuple): 归一化时的目标范围。
        Returns:
            torch.Tensor: 反归一化后的数据。
        """
        if feature_range != (0, 1):
            normalized_data = (normalized_data - feature_range[0]) / (feature_range[1] - feature_range[0])
        data = normalized_data * (max_val - min_val) + min_val
        return data


class CustomUNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 features=(16, 32, 64, 128, 256),  # 默认 features
                 act="gelu"):
        """
        自定义 UNet 初始化。
        :param in_channels: 输入通道数（图像通道数）
        :param out_channels: 输出通道数
        :param features: UNet 中的编码器和解码器特征层配置
        :param act: 激活函数
        """
        super(CustomUNet, self).__init__()
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.unet = BasicUNet(
            spatial_dims=2,
            features=features,
            act=act,
            in_channels=in_channels,  # 输入通道数加1以包含 mask
            out_channels=out_channels
        )

    def forward(self, x, mask = None):
        """
        前向传播函数。
        :param x: 输入图像 (batch, in_channels, height, width)
        :param mask: 输入 mask (batch, 1, height, width)
        :return: UNet 的输出
        """
        x = self.bn(x)                   # 先通过 BatchNorm2d
        # x = torch.cat((x, mask), dim=1)  # 在通道维度上拼接 mask
        x = self.unet(x)                 # 输入 UNet
        return x


class CustomMLP(nn.Module):
    def __init__(self, 
                 in_channels,
                 width,
                 height):
        """
        Args
            :param in_channels: 输入通道数（图像通道数）
            :param out_channels: 输出通道数
            :param features: UNet 中的编码器和解码器特征层配置
            :param act: 激活函数
        """
        super(CustomMLP, self).__init__()
        channels = in_channels * width * height
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.linear1 = nn.Linear(channels, in_channels)
        self.linear2 = nn.Linear(in_channels, channels)
        

    def forward(self, x):
        """
        Args
            :param x: 输入图像 (batch, in_channels, height, width)
            :param mask: 输入 mask (batch, 1, height, width)
            :return: UNet 的输出
        """
        tensor_size = x.size()
        x = self.bn(x)# 先通过 BatchNorm2d
        x = x.reshape(tensor_size[0],-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = torch.cat((x, mask), dim=1)  # 在通道维度上拼接 mask
        x = x.reshape(tensor_size)            # 输入 UNet
        return x


class CustomAutoencoder(nn.Module):
    def __init__(self, height, width, input_channels=3, latent_dim=128, downsampling_steps=2):
        """
        :param input_channels: 输入图像的通道数 (e.g., RGB=3, grayscale=1)
        :param latent_dim: 隐空间的维度大小
        :param downsampling_steps: 下采样的次数
        """
        super(CustomAutoencoder, self).__init__()
        self.downsampling_steps = downsampling_steps

        # Encoder: Dynamically add downsampling layers
        encoder_layers = []
        channels = input_channels
        for _ in range(downsampling_steps):
            encoder_layers.append(nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            channels *= 2
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent representation
        self.latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels*width//(2**self.downsampling_steps)*height//(2**self.downsampling_steps), latent_dim),  # Assuming minimum size reaches 4x4
            nn.ReLU(),
            nn.Linear(latent_dim, channels * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (channels, 4, 4))
        )
        
        # Decoder: Dynamically add upsampling layers
        decoder_layers = []
        for _ in range(downsampling_steps):
            decoder_layers.append(nn.ConvTranspose2d(channels, channels // 2, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU())
            channels //= 2
        decoder_layers[-1] = nn.ConvTranspose2d(channels, input_channels, kernel_size=4, stride=2, padding=1)
        decoder_layers.append(nn.Sigmoid())  # Normalize to [0, 1]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return x


class TinyImageNetValDataset(Dataset):
    def __init__(self, img_dir, annotations_file, class_to_idx, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        # 读取 val_annotations.txt 并解析标签
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_id = parts[1]
                label = class_to_idx[class_id]
                self.samples.append((img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class TinyImageNetResNet50(nn.Module):
    def __init__(self, num_classes=200, pretrained=False):
        super(TinyImageNetResNet50, self).__init__()
        # 创建 ResNet-50 模型
        self.base_model = models.resnet50(num_classes=num_classes, pretrained=pretrained)
        
        # 修改第一个卷积层（适配小图像）
        self.base_model.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # 移除最大池化层
        self.base_model.maxpool = nn.Identity()
        
        # 修改分类头
        # num_ftrs = self.base_model.fc.in_features
        # self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)
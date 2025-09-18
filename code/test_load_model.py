import resnet32
import trainer
import os
# import sys
import time
import zfpy
import wandb
import numpy as np
from tqdm import tqdm
# from pysz.pysz import SZ
from pympler import asizeof
# from nvidia import nvcomp
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from scipy.stats import entropy
# from scipy.spatial.distance import cosine
import math
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
# import code
# from concurrent.futures import ProcessPoolExecutor
from torchmetrics.functional import structural_similarity_index_measure as ssim
from cad_utils import generate_name,get_handle_front,check_active_block,check_freez_block,zfpy_compress_output,seperate_model,merge_models
from cad_dataset import CmpDataset,valDataset,CustomCmpBatchDataset,CustomAugmentedDataset,CustomUNet,CustomMLP,CustomTrainer,CustomAutoencoder,EncodableRandomHorizontalFlip,AddGaussianNoise

model = resnet32.resnet32(dataset='cifar100')
state_dict = torch.load('workDir/comp_and_drop/temp_model/model.th')
model.load_state_dict(state_dict["state_dict"])
model.cuda()

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2470, 0.2435, 0.2616])

val_loader = DataLoader(
        datasets.CIFAR100(root='/scratch/cy65664/workDir/comp_and_drop/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),download=True),
        batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)
criterion = nn.CrossEntropyLoss().cuda()
trainer.validate(val_loader, model, criterion)
import resnet32
import trainer
import os
# import sys
import math
import time
import zfpy_cuda
import wandb
import argparse
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

from torchmetrics.functional import structural_similarity_index_measure as ssim
from cad_utils import generate_name,get_handle_front,check_active_block,check_freez_block,zfpy_compress_output,seperate_model,merge_models
from cad_dataset import CmpDataset,valDataset,CustomCmpBatchDataset,CustomAugmentedDataset,CustomUNet,CustomMLP,CustomTrainer,CustomAutoencoder,EncodableRandomHorizontalFlip,AddGaussianNoise

torch.manual_seed(0)
np.random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

# subdir = f'Cmp4Train_exp/pytorch_resnet_cifar10/'
np_dir='./npdata/'

subdir = '/home/chence/workspace/Cmp4ai/comp_and_drop/'
img_dir = subdir+f'visualize_info/' 
# save_dir = os.path.join(workdir, 'transfer_exp', 'checkpoint')

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('-p', '--position',nargs = '+', default=None, type=int, metavar='N',
                    help='The position to register hook and freeze')
parser.add_argument('-cmp', '--compression', action='store_true',
                    help='if compress the activation')
parser.add_argument('-fhook', '--forward_hook', action='store_true',
                    help='if send the gradient information')
parser.add_argument('-drp', '--drop', nargs = '+', default=None, type=int, metavar='N',
                    help='if drop the previous layer')
parser.add_argument('-tol', '--tolerance', default=1e2, type=float, metavar='N',
                    help='the compression tolerance')
parser.add_argument('-gma', '--gamma', default=0, type=float, metavar='N',
                    help='the ratio of reserved channel')
parser.add_argument('-m', '--metric', default='ssim', type=str, choices=['psnr', 'ssim'],
                    help='type of metric to use for evaluation (default: psnr)')
parser.add_argument('-fzepo', '--freez_epoch', nargs = '+',type=int, metavar='N',
                    help='epochs to freeze')
parser.add_argument('-epo', '--epoch',default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cmp_batch_size', default=4, type=int, metavar='N',
                    help='the size of batch for compression')
parser.add_argument('-lm', '--learning_model',action='store_true',
                    help='if set the extra learning model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_gradients_list = []
weight_list = [0]*16

Output_list = []
best_prec1 = 0
compress_time = [[],[]]
compress_ratio = []
commu_cost = [[],[]]


unetdata = [[],[]]
dropstatue = False


def zfpy_compress_output(tol):
    def zfpy_cmp_inter(module, input, output):
        # global unetdata,Unet_model,Unet_train
        if module.training:
            # batch_size = output.shape[0]
            # sample_len = batch_size//2
            # cmpdata = output[:sample_len] # get the augmented data without compressed and decompressed
            # oridata = output[sample_len:]
           
            oridata = output.detach()

            trans_str= time.time()
            output = output.cpu().detach().numpy() # For training
            trans_end= time.time()
            intersize = output.nbytes
            t1 = time.time()
            compressed_data = zfpy.compress_numpy(output, tolerance=tol)
            t2 = time.time()
            intersize_cmpd = asizeof.asizeof(compressed_data)
            compress_ratio.append(intersize_cmpd/intersize)
            t3 = time.time()
            decompressed_array = zfpy.decompress_numpy(compressed_data)
            # decompressed_array_cal = decompressed_array.copy()
            t4 = time.time()
            wandb.log({f'cmp_ratio':intersize_cmpd/intersize})#f'error ratio':noise_ratio,f'cosine similarity':cos_sim}
            # plot_data_distribution(noise.flatten(),output.flatten(),decompressed_array.flatten(),f'error_distrbutuib\Err tol {parser.parse_args().tolerance} noise ratio {noise_ratio:.2e} eucl_dis {eucl_dis:.2e} cos {cos_sim:.2e}')
            
            trans2_sta = time.time()
            output_dec = torch.from_numpy(decompressed_array).to(device)

            trans2_end = time.time()
            # code.interact(local=locals())
            compress_time[0].append(t2 - t1)
            compress_time[1].append(t4 - t3)
            commu_cost[0].append(trans_end - trans_str)
            commu_cost[1].append(trans2_end - trans2_sta)
        # print('inter data cmp and decmp has completed')
            return output_dec
    return zfpy_cmp_inter


def get_unet_data(module, input, output):# augmentation only
    global unetdata,Unet_model,Unet_train
    output_unet = output.detach()
    if module.training:
        batch_size = output_unet.shape[0]
        sample_len = batch_size//2
        augdata = output_unet[:sample_len] # get the augmented data without compressed and decompressed
        oridata = output_unet[sample_len:]
        trans_mask = trainer.trans_mask()
        Unet_train.train(oridata,augdata,trans_mask)
        if Unet_train.run_batch == 1563:
            Unet_train.validation(oridata,augdata,trans_mask)
            Unet_train.run_batch = 0
            Unet_train.epoch += 1
    
        return output


def get_cmpdata(output, args):
    split_array = np.split(output, output.shape[0])
    shapes_list = [array.shape for array in split_array]
    if 5 in args.drop:
        all_shapes_consistent = all(shape == (1, 32, 32, 32) for shape in shapes_list)
    elif 10 in args.drop:
        all_shapes_consistent = all(shape == (1, 64, 16, 16) for shape in shapes_list)
    assert all_shapes_consistent, "Not all shapes are consistent"
    compressed_list = [zfpy.compress_numpy(array, tolerance=args.tolerance) for array in split_array]
    decompress_list = [zfpy.decompress_numpy(array) for array in compressed_list]
    combined_dcmp = np.concatenate(decompress_list, axis=0)  # 按行拼接
    dcmp_output = torch.from_numpy(combined_dcmp).to(device)
    return dcmp_output

def compute_ssim_per_channel(x, y):
    # x 和 y 形状: [B, C, H, W]
    batch_size, channels, _, _ = x.shape
    ssim_values = []
    for c in range(channels):
        ssim_value = ssim(x[:, c:c+1, :, :], y[:, c:c+1, :, :], data_range=1.0)
        ssim_values.append(ssim_value.item())
    return ssim_values

def compute_mse_psnr_per_channel(x, y, data_range=1.0):
    # x 和 y 形状: [B, C, H, W]
    batch_size, channels, _, _ = x.shape
    mse_loss = torch.nn.MSELoss(reduction='mean')

    mse_values = []
    psnr_values = []

    for c in range(channels):
        mse = mse_loss(x[:, c, :, :], y[:, c, :, :])
        mse_values.append(mse.item())

        # 通过 MSE 计算 PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * math.log10(data_range) - 10 * math.log10(mse.item())
        psnr_values.append(psnr)

    return mse_values, psnr_values

def get_top_n_channels(values, n=3):
    # 对索引和值进行排序，按照值的降序排列
    sorted_indices = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
    # 取前 n 个的索引
    top_n_indices = [index for index, value in sorted_indices[:n]]
    return top_n_indices

# 获取最小的 N 个通道索引
def get_bottom_n_channels(values, n=3):
    # 对索引和值进行排序，按照值的升序排列
    sorted_indices = sorted(enumerate(values), key=lambda x: x[1])
    # 取前 n 个的索引
    bottom_n_indices = [index for index, value in sorted_indices[:n]]
    return bottom_n_indices


def infer_original_channel_count(output_tensor, num_aug_versions=5):
    """
    从模型输出 tensor 推断原始通道数 C。

    参数:
        output_tensor (Tensor): 模型输出，形状为 (B, (1+num_aug_versions)*C, H, W)
        num_aug_versions (int): 增强版本数量（不含原图）

    返回:
        int: 原始图像的通道数 C
    """
    total_channels = output_tensor.shape[1]
    total_parts = 1 + num_aug_versions
    assert total_channels % total_parts == 0, "输出通道数无法整除增强版本数+1"
    return total_channels // total_parts


def main():
    global args, best_prec1, Unet_model,Unet_train
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2470, 0.2435, 0.2616])
    ## Normal Data
    cifar10_data = datasets.CIFAR10(root='/home/chence/workspace/Cmp4ai/comp_and_drop/data', train=True, download=True)
    cifar100_data = datasets.CIFAR100(root='/home/chence/workspace/Cmp4ai/comp_and_drop/data', train=True, download=True)
    val_loader = DataLoader(
        datasets.CIFAR100(root='/home/chence/workspace/Cmp4ai/comp_and_drop/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),download=True),
        batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)
    
    

    train_loader = DataLoader(
        datasets.CIFAR100(root='/home/chence/workspace/Cmp4ai/comp_and_drop/data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size = 64, shuffle=True, #args.batch_size
        num_workers = 4, pin_memory=True)

    # subdir = f'Cmp4Train_exp/pytorch_resnet_cifar10/'
    save_dir = subdir + 'temp_model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    start_t=time.time()


    

    # -------------------
    # Define Resnet model
    # -------------------
    model = resnet32.resnet32(dataset='cifar100')
    # state_dict = torch.load('workDir/comp_and_drop/temp_model/model.th')
    # model.load_state_dict(state_dict["state_dict"],strict=False)
    # num_ftrs = model.linear.in_features
    # model.linear = nn.Linear(num_ftrs, 100) 
    model=model.to("cuda")
    model.cuda()
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-6, last_epoch=-1)
 
    # gradients = []


    wandb.init(
        # set the wandb project where this run will be logged
        project="Comp And Drop resnet32 rebuttal",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.1,
        "architecture": "Resnet32",
        "dataset": "CIFAR-100",
        "epochs": args.epoch,
        "compression tolerance": args.tolerance,
        },
        name = generate_name(args),
        notes = f"batch size {args.cmp_batch_size} compression proposed Flip with {args.metric} gamma {args.gamma}",
    )

    

    trainer.validate(val_loader, model, criterion)
    check_active_block(model,0) # activate all the blocks
    
    epoch_num = args.epoch
    drop =[False]*16
    freez = [False]*16
    frz_flag = drop_flag = 0
    handle_list = []
    cmp_val = [[],[]]
    dropstatue = False
    cmpstatue = False
    front_model = None
    tol_exist = False


   
    i = 0

    for epoch in range(0, epoch_num):
        
        if hasattr(args, 'freez_epoch') and args.freez_epoch:
            if epoch in args.freez_epoch:
                if args.position is not None and not freez[args.position[frz_flag]]:
                    if args.compression:
                        handle = get_handle_front(model,args.position[frz_flag])
                        if handle is not None:
                            hook_handle_f = handle.register_forward_hook(zfpy_compress_output(args.tolerance)) #register compression through hook
                            handle_list.append(hook_handle_f)
                        else:
                            raise ValueError(f"Unable to retrieve valid layer for pos {args.position[frz_flag]}.")
                        cmpstatue = True
                    
                    check_freez_block(model,args.position[frz_flag]) # freeze the conv layer not bn layer
                    check_active_block(model,args.position[frz_flag]+1) # unfreeze the rest block
                    drop[frz_flag] = True
                    frz_flag += 1
        
                if hasattr(args, 'drop') and args.drop:
                    if not drop[args.drop[drop_flag]]:
                        if front_model is not None:
                            model_templete = resnet32.resnet32(dataset='cifar100')
                            num_ftrs = model_templete.linear.in_features
                            model_templete.linear = nn.Linear(num_ftrs, 100) 
                            model = merge_models(front_model,model,model_templete,key_mapping)
                            model = model.to("cuda")
                        front_model,remaining_model,key_mapping = seperate_model(model,args.drop[drop_flag])
                        cmp_act=[[],[]]
                    for param in front_model.parameters(): 
                        param.requires_grad=False
                        param.grad = None
                    front_model.eval()

                    # -------------------
                    # define the training data of custom model
                    # -------------------
                    # augmented_dataset = AugmentedDataset(cifar10_data, transform=data_aug)
                    # train_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
                    post_transforms = transforms.Compose([normalize]) # ,transforms.ToTensor()
                    cifar100_data = datasets.CIFAR100(root='/home/chence/workspace/Cmp4ai/comp_and_drop/data', train=True, download=True) # ,transform=post_transforms
                    augmented_dataset = CustomAugmentedDataset(data=cifar100_data,
                                                                flip_prob=1,
                                                                transform=post_transforms
                                                            )
                    
                    custom_train_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=False)
                    if 5 >= args.drop[drop_flag]:
                        channels = 32
                    elif 10 >= args.drop[drop_flag]:
                        channels = 64


                    gamma = args.gamma
                    reamin_num = round(channels*gamma)
                    ssim_list = []
                    mse_list = []
                    psnr_list = []
                    #     # -------------------
                    #     # Prepare the selection vector for the training data
                    for i, batch in enumerate(custom_train_loader):

                        if len(batch) == 4:
                            augmented_batch, original_batch, labels, mask = batch
                            # input = torch.cat((augmented_batch, original_batch), dim=0)  # 在第0维（batch 维度）拼接
                            B, N_aug, C, H, W = augmented_batch.shape
                            aug_input = augmented_batch.view(B * N_aug, C, H, W)  # (5B, C, H, W)
                            ori_input = original_batch                     # (B, C, H, W)
                            input = torch.cat([aug_input, ori_input], dim=0)
                            target = torch.cat([labels.repeat(N_aug), labels], dim=0) # 拼接后的标签，重复原始标签
                        elif len(batch) == 2:
                            input, target = batch

                        # 准备intermediate data 做为 training data
                        with torch.no_grad():
                            input_var = input.cuda()
                            output = front_model(input_var)

                        aug_output = output[:B * N_aug]    # (5B, total_channels, H, W)
                        ori_output = output[B * N_aug:]    # (B, total_channels, H, W)
                        # C_out = infer_original_channel_count(output, num_aug_versions=N_aug)
        # (5B, C, H, W)
                        aug_ori_feat = aug_output.view(B, N_aug, -1, H, W)

                        flipped_data = aug_ori_feat[:, 0, :, :, :]          # (B, C, H, W)，增强版本0 = flip
              # (B, C, H, W)
                        flipped_ori_data = torch.flip(ori_output, dims=[3])  # 左右翻原图特征

                        channel_ssim_temp = compute_ssim_per_channel(flipped_data, flipped_ori_data)
                        channel_mse_temp,channel_psnr_temp = compute_mse_psnr_per_channel(flipped_data, flipped_ori_data)
                        ssim_list.append(channel_ssim_temp)
                        mse_list.append(channel_mse_temp)
                        psnr_list.append(channel_psnr_temp)
                        if i >100:
                            break
                    channel_ssim = np.mean(ssim_list, axis=0)
                    channel_psnr = np.mean(psnr_list, axis=0)
                    psnr_index = get_bottom_n_channels(channel_psnr, n=reamin_num)
                    ssim_index = get_bottom_n_channels(channel_ssim, n=reamin_num)

                    # -----------------------------------------------
                    # Prepare the training tensor for resnet
                    # -----------------------------------------------
                    custom_train_loader = DataLoader(augmented_dataset, batch_size=args.cmp_batch_size, shuffle=False)
                    start_storage= time.time()
                    data_size = 0
                    tensor_size = 0
                    cmp_data_size = 0
                    
                    for j, batch in enumerate(custom_train_loader):
                        if len(batch) == 4:
                            augmented_batch, original_batch, labels, mask = batch
                            B, N_aug, C, H, W = augmented_batch.shape
                            aug_input = augmented_batch.view(B * N_aug, C, H, W)  # (5B, C, H, W)
                            input = torch.cat((aug_input, original_batch), dim=0)  # 在第0维（batch 维度）拼接
                            target = labels # torch.cat([labels.repeat(N_aug), labels], dim=0) #torch.cat((labels, labels), dim=0) # 拼接后的标签，重复原始标签
                        elif len(batch) == 2:
                            input, target = batch

                            # 准备intermediate data 做为 training data
                        with torch.no_grad():
                            input_var = input.cuda()
                            output = front_model(input_var)

                        # -------------------------------------------------------------------
                        # Get the augmentation layer and concatenate to the original tensor
                        # -------------------------------------------------------------------
                        if args.metric == 'ssim':
                            select_index = ssim_index
                        elif args.metric == 'psnr':
                            select_index = psnr_index

                        aug_tensor = output[:B * N_aug]         # shape: (5B, C, H, W)
                        ori_tensor = output[B * N_aug:]         # shape: (B, C, H, W)

                        # reshape 成 (B, N_aug, C, H, W)
                        aug_tensor = aug_tensor.view(B, N_aug, -1, H, W)

                        # 提取每个增强版本的敏感通道
                        reserved_aug_data = aug_tensor[:, :, psnr_index, :, :]  # shape: (B, 5, len, H, W)

                        # reshape 成 (B, 5 * len, H, W)
                        reserved_aug_data = reserved_aug_data.view(B, -1, H, W)

                        # 拼接
                        cmp_data = torch.cat((ori_tensor, reserved_aug_data), dim=1)

                        # aug_tensor = output[:len(output)//2]
                        # select_index = []
                        # for aug_code in [0, 1, 2, 3, 4]:
                        #     offset = aug_code * C
                        #     select_index.extend([i + offset for i in psnr_index])

                        # reserved_aug_data = aug_tensor[:,select_index,:,:]
                        # cmp_data = torch.cat(( output[len(output)//2:], reserved_aug_data), dim=1)
                        cmp_data = cmp_data.cpu().detach().numpy() 
                        intersize = cmp_data.nbytes
                        tensor_size += intersize
                        
                        if not tol_exist:
                            tol_rate = zfpy_cuda.compute_rate_from_tolerance(cmp_data,args.tolerance)
                        cmpd_data = zfpy_cuda.compress_tensor(cmp_data, rate_bits_per_value=tol_rate)
                        cmpd_size = asizeof.asizeof(cmpd_data)
                        
                        
                        

                        # split_array = np.split(cmp_data, cmp_data.shape[0])
                        # compressed_list = [zfpy.compress_numpy(array, tolerance=args.tolerance) for array in split_array]
                        # cmpd_size = asizeof.asizeof(compressed_list)
                        data_size += cmpd_size
                        compress_ratio.append(cmpd_size/intersize)
                        wandb.log({f'cmp_ratio':cmpd_size/intersize,'cmp_data_size':cmpd_size})

                        # shapes_list = [array.shape for array in split_array]
                        # if 5 >= args.drop[drop_flag]:
                        #     channel_num = 32 + reamin_num
                        #     all_shapes_consistent = all(shape == (1, channel_num, 32, 32) for shape in shapes_list)
                        # elif 10 >= args.drop[drop_flag]:
                        #     channel_num = 64 + reamin_num
                        #     all_shapes_consistent = all(shape == (1, channel_num, 16, 16) for shape in shapes_list)
                        # assert all_shapes_consistent, "Not all shapes are consistent"


                        #-----------------------------------------------
                        # Directly Flip Baseline
                        # train_loader_cmp = DataLoader(datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                        #                                 transforms.ToTensor(),
                        #                                 normalize,
                        #                                 ]), download=True), batch_size=64, shuffle=True)
                        # start_storage= time.time()
                        # tensor_size = cmp_data_size = 0
                        # for j,(input, target) in enumerate(train_loader_cmp):
                        #     # 准备intermediate data 做为 training data
                        #     with torch.no_grad():
                        #         input_var = input.cuda()
                        #         output = front_model(input_var)
                        #     output = output.cpu().detach().numpy() # For training
                        #     tensor_size += output.nbytes
                        #     intersize = output.nbytes
                        #     # cmpd_data = zfpy.compress_numpy(output, tolerance=args.tolerance[drop_flag])
                            
                            
                        #        # compress the data individually 
                        #     split_array = np.split(output, output.shape[0])
                        #     # train_size += sum(asizeof.asizeof(array) for array in split_array)
                        #     shapes_list = [array.shape for array in split_array]

                            # if 5 >= args.drop[drop_flag]:
                            #     all_shapes_consistent = all(shape == (1, 32, 32, 32) for shape in shapes_list)
                            # elif 10 >= args.drop[drop_flag]:
                            #     all_shapes_consistent = all(shape == (1, 64, 16, 16) for shape in shapes_list)
                            # assert all_shapes_consistent, "Not all shapes are consistent"
                        #     # method end
                        #     #-----------------------------------------------


                        
                        # cmp_act[0].extend(compressed_list)
                        # cmp_act[1].extend(target)
                        cmp_act[0].append(cmpd_data)
                        cmp_act[1].append(target)
                        if j % 50 == 0:#args.print_freq == 0:
                            print('Batch: [{0}/{1}]\t activation data has been saved '.format(j, len(custom_train_loader)))

                    assert len(cmp_act[0])==len(cmp_act[1]),"The length of the compressed data and target data is not equal"
                    end_storage = time.time()
                    print(f'The storage cost {end_storage-start_storage}s') 
                    print(f'The size of the training data tensor data is {data_size/1024**2:.2f} MB')
                    wandb.log({f'cmp_ratio':data_size/tensor_size,'cmp_data_size':data_size/1024**2})   

                    if 5 >= args.drop[drop_flag]:
                        crop_transform = transforms.RandomCrop(32, 4, padding_mode='constant')
                        # rota_transform = transforms.RandomRotation(degrees=20)
                        blur_transform = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.5))
                        data_augmentation = transforms.Compose([
                        # transforms.RandomHorizontalFlip(p=0.5),
                        AddGaussianNoise(0, 0.03),
                        # rota_transform,
                        # blur_transform,
                        crop_transform
                        ])
                    elif 10 >= args.drop[drop_flag]:
                        crop_transform = transforms.RandomCrop(16, 2, padding_mode='constant')
                        # rota_transform = transforms.RandomRotation(degrees=40)
                        blur_transform = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
                        data_augmentation = transforms.Compose([
                        # transforms.RandomHorizontalFlip(p=0.5),
                        AddGaussianNoise(0, 0.01),
                        # rota_transform,
                        blur_transform,
                        crop_transform
                        ])
                        rota_transform = None

                    train_dataset = CustomCmpBatchDataset(cmp_act[0], cmp_act[1], flip_prob=0.5, transform=data_augmentation
                                                          ,ori_tensor_channel=channels
                                                          ,aug_index=select_index) # 使用 DataLoader 封装数据集

                    # # 使用 DataLoader 封装数据集
                    # train_dataset = CmpDataset(cmp_act[0], cmp_act[1], transform=data_augmentation) # 使用 DataLoader 封装数据集
                    # val_dataset = valDataset(cmp_val[0], cmp_val[1], transform=None)

                    # 使用 DataLoader 封装数据集
                    batch_size = 64//args.cmp_batch_size
                    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 4, pin_memory=True)




                




                    model = remaining_model
                    # 重新定义optimizer和lr_scheduler
                    front_model = front_model.to("cpu")
                    currentlr = optimizer.param_groups[0]['lr']
                    optimizer = torch.optim.SGD(model.parameters(), currentlr,
                                        momentum=0.9,
                                        weight_decay=1e-4)
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch-epoch, eta_min=1e-6, last_epoch=-1)
                    dropstatue = True
                    drop_flag += 1
                    model = model.to("cuda")
                    model.cuda()

                
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        trainer.train(train_loader, model, criterion, optimizer, epoch, args, wandb)
    
        lr_scheduler.step()

        # evaluate on validation set
        if front_model is not None:
            front_model = front_model.to("cuda")
            model_templete = resnet32.resnet32(dataset='cifar100')
            num_ftrs = model_templete.linear.in_features
            model_templete.linear = nn.Linear(num_ftrs, 100) 
            mergemodel = merge_models(front_model,model,model_templete,key_mapping)
            mergemodel = mergemodel.to("cuda")
            mergemodel.cuda()
            prec1 = trainer.validate(val_loader, mergemodel, criterion)
        # # run["validate/acc"].append(prec1)
        else:
            prec1 = trainer.validate(val_loader, model, criterion)
            
        wandb.log({"acc": prec1})

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % 10 == 0:
            trainer.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(save_dir, 'checkpoint_transfer.th'))
        if is_best:
            if dropstatue:
                model_templete = resnet32.resnet32(dataset='cifar100')
                num_ftrs = model_templete.linear.in_features
                model_templete.linear = nn.Linear(num_ftrs, 100) 
                combine_model = merge_models(front_model,model,model_templete,key_mapping)
                model_save = combine_model
            else:
                model_save=model

            trainer.save_checkpoint({
                'state_dict': model_save.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(save_dir, 'model_transfer.th'))




    
    end_t = time.time()
    duration = end_t - start_t
    print(f'The training cost {duration}s')
    ave_ratio=np.mean(compress_ratio)
    ave_time = np.mean(compress_time,1)
    ave_cmu = np.mean(commu_cost,1)
    if args.compression:
        print(f'The average compression ratio {ave_ratio:.4f}')
        print(f'The average compression time {ave_time[0]:.4f}, and the decompression time {ave_time[1]:.4f}')
        print(f'The average cuda2cpu cost {ave_cmu[0]:.4f}, and the cpu2cuda cost {ave_cmu[1]:.4f}')
        wandb.log({"Average compression ratio":ave_ratio,"compression time":ave_time[0],"decompression time":ave_time[1],"cuda2cpu cost":ave_cmu[0],"cpu2cuda cost":ave_cmu[1]})
    wandb.finish()
    for handle in handle_list:
        handle.remove()


if __name__ == '__main__':
    main()
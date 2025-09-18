import resnet32
import resnet32_2
import trainer
import os
import sys
import time
import zfpy
import wandb
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pysz.pysz import SZ
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
from scipy.spatial.distance import cosine
import torch.nn.functional as F
import multiprocessing
import code
from concurrent.futures import ProcessPoolExecutor
from cad_utils import generate_name,get_handle_front,check_active_block,check_freez_block,zfpy_compress_output,seperate_model,merge_models
from cad_dataset import CmpDataset,valDataset,AddGaussianNoise,CmpBatchDataset

torch.manual_seed(42)

'''
Version note:
     this version has the complete code for drop the front part and build the tensor as the dataset
     the model will be seperated into two parts, the front part will be freezed and the back part will be trained
     the front part will be used to generate the intermediate data for the training
'''

subdir = f'Cmp4Train_exp/pytorch_resnet_cifar10/'
np_dir='./npdata/'
img_dir = subdir+f'visualize_info/' 

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('-p', '--position',nargs = '+', default=None, type=int, metavar='N',
                    help='The position to register hook and freeze')
parser.add_argument('-cmp', '--compression', action='store_true',
                    help='if compress the activation')
parser.add_argument('-fhook', '--forward_hook', action='store_true',
                    help='if compress the activation')
parser.add_argument('-drp', '--drop', nargs = '+', default=None, type=int, metavar='N',
                    help='if drop the previous layer')
parser.add_argument('-tol', '--tolerance', nargs = '+', default=1e-3, type=float, metavar='N',
                    help='the decompression tolerance')
parser.add_argument('-fzepo', '--freez_epoch', nargs = '+',type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-epo', '--epoch',default=160, type=int, metavar='N',
                    help='number of total epochs to run')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_gradients_list = []
weight_list = [0]*16

Output_list = []
best_prec1 = 0
compress_time = [[],[]]
compress_ratio = []
commu_cost = [[],[]]

cmp_act=[[],[]]
dropstatue = False



def zfpy_compress_output(tol):
    def zfpy_cmp_inter(module, input, output):
        if module.training:
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
            # act_vle = np.mean(np.abs(output))
            # noise = decompressed_array_cal - output
            # noise_mean = np.mean(np.abs(noise))
            # noise_ratio = noise_mean/act_vle
            # cos_sim = 1 - cosine(output.flatten(), decompressed_array_cal.flatten())
            wandb.log({f'cmp_ratio':intersize_cmpd/intersize})
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

def zfpy_compress_input_for(tol):
    def zfpy_cmp_inter(module, input, output):
        if module.training:
            trans_str= time.time()
            input = input.cpu().detach().numpy() # For training
            trans_end= time.time()
            intersize =input.nbytes
            t1 = time.time()
            compressed_data = zfpy.compress_numpy(input, tolerance=tol)
            # cmp_act[0].append(compressed_data)
            t2 = time.time()
            intersize_cmpd = asizeof.asizeof(compressed_data)
            compress_ratio.append(intersize_cmpd/intersize)
            t3 = time.time()
            decompressed_array = zfpy.decompress_numpy(compressed_data)
            decompressed_array_cal = decompressed_array.copy()
            t4 = time.time()
            act_vle = np.mean(np.abs(input))
            noise = decompressed_array_cal - input
            noise_mean = np.mean(np.abs(noise))
            noise_ratio = noise_mean/act_vle
            cos_sim = 1 - cosine(input.flatten(), decompressed_array_cal.flatten())
            wandb.log({f'cmp_ratio':intersize_cmpd/intersize,f'error ratio':noise_ratio,f'cosine similarity':cos_sim})
            # plot_data_distribution(noise.flatten(),output.flatten(),decompressed_array.flatten(),f'error_distrbutuib\Err tol {parser.parse_args().tolerance} noise ratio {noise_ratio:.2e} eucl_dis {eucl_dis:.2e} cos {cos_sim:.2e}')
            
            trans2_sta = time.time()
            input_dec = torch.from_numpy(decompressed_array).to(device)
            trans2_end = time.time()
            # code.interact(local=locals())
            compress_time[0].append(t2 - t1)
            compress_time[1].append(t4 - t3)
            commu_cost[0].append(trans_end - trans_str)
            commu_cost[1].append(trans2_end - trans2_sta)
        # print('inter data cmp and decmp has completed')
            return input_dec
    return zfpy_cmp_inter


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


def main():
    global args, best_prec1
    args = parser.parse_args()
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2470, 0.2435, 0.2616])
    ## Normal Data
    train_loader = DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size = 64, shuffle=True, #args.batch_size
        num_workers = 4, pin_memory=True)

    val_loader = DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)
    data_size = 0
    cifar10 = datasets.CIFAR10(root='./data', train=True, transform=None)
    for i in range(len(cifar10)):
        sample = cifar10[i][0]  # 获取单个样本
        data_size += asizeof.asizeof(sample)  # 测量样本大小
    data_size_mb = data_size / (1024 ** 2)
    print(f"Dataset size in memory: {data_size_mb:.2f} MB")

    
    


    subdir = f'Cmp4Train_exp/pytorch_resnet_cifar10/'
    save_dir = subdir + 'temp_model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    start_t=time.time()

    model = resnet32.resnet32()
    model=model.to("cuda")

    # model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    # register Hook function
    model.cuda()
    fwd_handle_list = []
    forward_flag = 0

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                     milestones=[100, 160, 200], last_epoch=0 - 1)
    # lr_scheduler = CubicCosineAnnealingLR(optimizer, args.epoch, eta_min=1e-4, last_epoch=-1, gamma=3.0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 160, eta_min=1e-4, last_epoch=-1)
    # gradients = []

    

    trainer.validate(val_loader, model, criterion)
    check_active_block(model,0) # activate all the blocks
    
    epoch_num = args.epoch
    freez = drop = [False]*16
    frz_flag = drop_flag = 0
    handle_list = []
    
    dropstatue = False
    front_model = None


    wandb.init(
        # set the wandb project where this run will be logged
        project="Comp And Drop epoch batch compression test",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.1,
        "architecture": "Resnet32",
        "dataset": "CIFAR-10",
        "epochs": epoch_num,
        "compression tolerance": args.tolerance,
        },
        name = generate_name(args),
        notes = "individual compression with modify merge model (remove the relu after first bn) validation",
    )
    i = 0

    for epoch in range(0, epoch_num):
        
        if hasattr(args, 'freez_epoch') and args.freez_epoch:
            if epoch in args.freez_epoch:
                if args.position is not None and not freez[args.position[frz_flag]]:
                    if args.compression:
                        if handle_list is not None:
                            for handle in handle_list:
                                handle.remove()
                            handle_list = []
                        handle = get_handle_front(model,args.position[frz_flag])
                        if handle is not None:
                            hook_handle_f = handle.register_forward_hook(zfpy_compress_output(args.tolerance[frz_flag])) #register compression through hook
                            handle_list.append(hook_handle_f)
                        else:
                            raise ValueError(f"Unable to retrieve valid layer for pos {args.position[frz_flag]}.")

                    check_freez_block(model,args.position[frz_flag]) # freeze the conv layer not bn layer
                    check_active_block(model,args.position[frz_flag]+1) # unfreeze the rest block
                    freez[args.position[frz_flag]] = True
                    frz_flag += 1


                if hasattr(args, 'drop') and args.drop:
                    if not drop[args.drop[drop_flag]]:
                        if front_model is not None:
                            model = merge_models(front_model,model,resnet32_2.resnet32(),key_mapping)
                            model = model.to("cuda")
                        front_model,remaining_model,key_mapping = seperate_model(model,args.drop[drop_flag])
                        cmp_act = [[],[]]
                        # cmp_val = [[],[]]
                        drop[args.drop[drop_flag]] = True
                        
                    # model.to("cpu")
                    for param in front_model.parameters(): 
                        param.requires_grad=False
                        param.grad = None
                    front_model.eval()
                    train_loader_cmp = DataLoader(datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    normalize,
                                                    ]), download=True), batch_size=64, shuffle=True)
                    # val_loader_cmp = DataLoader(datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                    #                                 transforms.ToTensor(),
                    #                                 normalize,
                    #                                 ]), download=True), batch_size=16, shuffle=True)
                    start_storage= time.time()
                    tensor_size = cmp_data_size = 0
                    for j,(input, target) in enumerate(train_loader_cmp):
                        # 准备intermediate data 做为 training data
                        with torch.no_grad():
                            input_var = input.cuda()
                            output = front_model(input_var)
                        output = output.cpu().detach().numpy() # For training
                        tensor_size += output.nbytes
                        intersize = output.nbytes
                        # cmpd_data = zfpy.compress_numpy(output, tolerance=args.tolerance[drop_flag])
                        
                        
                           # compress the data individually 
                        split_array = np.split(output, output.shape[0])
                        # train_size += sum(asizeof.asizeof(array) for array in split_array)
                        shapes_list = [array.shape for array in split_array]

                        if 5 >= args.drop[drop_flag]:
                            all_shapes_consistent = all(shape == (1, 32, 32, 32) for shape in shapes_list)
                        elif 10 >= args.drop[drop_flag]:
                            all_shapes_consistent = all(shape == (1, 64, 16, 16) for shape in shapes_list)
                        assert all_shapes_consistent, "Not all shapes are consistent"

                        compressed_list = [zfpy.compress_numpy(array, tolerance=args.tolerance[drop_flag]) for array in split_array]

                        #----------------------------------------------------------
                        # test the data size and compression ratio

                            # individually compress
                        intersize_cmpd = asizeof.asizeof(compressed_list)
                        cmp_data_size += sum(asizeof.asizeof(block) for block in compressed_list)
                        wandb.log({f'cmp_ratio':intersize_cmpd/intersize})

                            # batchly compress
                        # cmpd_size = asizeof.asizeof(cmpd_data)
                        # cmp_data_size += cmpd_size
                        # wandb.log({f'cmp_ratio':cmpd_size/intersize})

                        compress_ratio.append(intersize_cmpd/intersize)
                        
                        
                        # compressed_data = zfpy.compress_numpy(output, tolerance=args.tolerance)
                        #----------------------------------------------------------

                        cmp_act[0].extend(compressed_list)
                        cmp_act[1].extend(target)
                        # cmp_act[0].append(cmpd_data)
                        # cmp_act[1].append(target)
                        if j % 50 == 0:#args.print_freq == 0:
                            print('Batch: [{0}/{1}]\t activation data has been saved '.format(j, len(train_loader_cmp)))
                        # compute gradient and do SGD step
                    cmpd_data_size_v = asizeof.asizeof(cmp_act[0])
                    print(f'The size of the training data tensor data is {cmpd_data_size_v/1024**2:.2f} MB')
                    
                    assert len(cmp_act[0])==len(cmp_act[1]),"The length of the compressed data and target data is not equal"
                    # assert len(cmp_act[0])==len(train_loader_cmp),"The compressed data has different length with the train_loader"



                    # for j,(input, target) in enumerate(val_loader_cmp):
                    #     # 准备intermediate data 做为 training data
                    #     with torch.no_grad():
                    #         input_var = input.cuda()
                    #         output = front_model(input_var)
                    #     output = output.cpu().detach().numpy() # For training
                    #     tensor_size += output.nbytes
                    #     intersize = output.nbytes
                        
                        
                    #        # compress the data individually 
                    #     split_array = np.split(output, output.shape[0])

                    #     shapes_list = [array.shape for array in split_array]

                    #     if 5 >= args.drop[drop_flag]:
                    #         all_shapes_consistent = all(shape == (1, 32, 32, 32) for shape in shapes_list)
                    #     elif 10 >= args.drop[drop_flag]:
                    #         all_shapes_consistent = all(shape == (1, 64, 16, 16) for shape in shapes_list)
                    #     assert all_shapes_consistent, "Not all shapes are consistent"

                    #     # compressed_list = [zfpy.compress_numpy(array, tolerance=args.tolerance[drop_flag]) for array in split_array]

                    #     #----------------------------------------------------------
                    #     # test the data size and compression ratio

                    #         # individually compress
                    #     # intersize_cmpd = asizeof.asizeof(compressed_list)
                    #     # cmp_data_size += sum(asizeof.asizeof(block) for block in compressed_list)
                    #     # wandb.log({f'cmp_ratio':intersize_cmpd/intersize})

                    #         # batchly compress
                    #     # cmpd_size = asizeof.asizeof(cmpd_data)
                    #     # cmp_data_size += cmpd_size
                    #     # wandb.log({f'cmp_ratio':cmpd_size/intersize})

                    #     compress_ratio.append(intersize_cmpd/intersize)
                        
                        
                    #     # compressed_data = zfpy.compress_numpy(output, tolerance=args.tolerance)
                    #     #----------------------------------------------------------

                    #     cmp_val[0].extend(split_array)
                    #     cmp_val[1].extend(target)
                    #     # cmp_act[0].append(cmpd_data)
                    #     # cmp_act[1].append(target)
                    #     if j % 50 == 0:#args.print_freq == 0:
                    #         print('Batch: [{0}/{1}]\t activation data has been saved '.format(j, len(train_loader_cmp)))
                    #     # compute gradient and do SGD step
                    end_storage = time.time()
                    print(f'The storage cost {end_storage-start_storage:.2f}s')


                    if 5 >= args.drop[drop_flag]:
                        crop_transform = transforms.RandomCrop(32, 4, padding_mode='constant')
                        rota_transform = transforms.RandomRotation(degrees=20)
                        blur_transform = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.5))
                        data_augmentation = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        AddGaussianNoise(0, 0.03),
                        # rota_transform,
                        # blur_transform,
                        crop_transform
                        ])
                        # rota_transform = None
                    elif 10 >= args.drop[drop_flag]:
                        crop_transform = transforms.RandomCrop(16, 2, padding_mode='constant')
                        rota_transform = transforms.RandomRotation(degrees=40)
                        blur_transform = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
                        data_augmentation = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        AddGaussianNoise(0, 0.01),
                        # rota_transform,
                        blur_transform,
                        crop_transform
                        ])
                    train_dataset = CmpDataset(cmp_act[0], cmp_act[1], transform=data_augmentation) # 使用 DataLoader 封装数据集
                    # val_dataset = valDataset(cmp_val[0], cmp_val[1], transform=None)

                    # 使用 DataLoader 封装数据集
                    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True, num_workers = 4, pin_memory=True)
                    # val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=True, num_workers = 4, pin_memory=True)
                    model = remaining_model
                    # 重新定义optimizer和lr_scheduler
                    currentlr = optimizer.param_groups[0]['lr']
                    optimizer = torch.optim.SGD(model.parameters(), currentlr,
                                        momentum=0.9,
                                        weight_decay=1e-4)
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160-epoch, eta_min=1e-4, last_epoch=-1)
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
            mergemodel = merge_models(front_model,model,resnet32_2.resnet32(),key_mapping)
            mergemodel = mergemodel.to("cuda")
            # mergemodel.cuda()
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
            }, is_best, filename=os.path.join(save_dir, 'checkpoint.th'))
        if is_best:
            if dropstatue:
                combine_model = merge_models(front_model,model,resnet32.resnet32(),key_mapping)
                model_save = combine_model
            else:
                model_save=model

        trainer.save_checkpoint({
            'state_dict': model_save.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(save_dir, 'model.th'))




    
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
    # multiprocessing.set_start_method('spawn')
    main()
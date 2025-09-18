import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import time
import code
import wandb
import zfpy
import copy
import torch.nn as nn
import torch.nn.functional as F

subdir = f'Cmp4Train_exp/pytorch_resnet_cifar10/'
np_dir='./npdata/'
img_dir = subdir+f'visualize_info/' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight_gradients_list = []
weight_list = [0]*16
Output_list = []
best_prec1 = 0
compress_time = [[],[]]
compress_ratio = []
commu_cost = [[],[]]
weight_dict = {}
weight_val_dict = {}

def generate_name(args):
    name = f'epo_{args.epoch}'

    if hasattr(args, 'position') and args.position:
        # name += '_fz_p'
        name += f'_fz_p_{args.position}'

    if hasattr(args, 'drop') and args.drop:
        # name += '_drp'
        name += f'_drp_{args.drop}'
        if hasattr(args, 'tolerance') and args.tolerance:
               name += f'_tol_{args.tolerance}'
        if hasattr(args, 'gamma') and args.gamma:
               name += f'_gma_{args.gamma}'
        if hasattr(args, 'metric') and args.metric:
               name += f'_m_{args.metric}'
        if hasattr(args, 'cmp_batch_size') and args.cmp_batch_size:
               name += f'_cmp_b_size_{args.cmp_batch_size}'

    if hasattr(args, 'forward_hook') and args.forward_hook:
        name += '_fhook'

    if hasattr(args, 'freez_epoch') and args.freez_epoch:
        # name += '_fz_epo'
        name += f'_fz_epo_{args.freez_epoch}'

    if hasattr(args, 'compression') and args.compression:
        name += f'_cmp_{args.compression}'
        if hasattr(args, 'tolerance') and args.tolerance:
            name += f'_tol_{args.tolerance}'
        

    if hasattr(args, 'learning_model') and args.learning_model:
            name += f'_lm_{args.learning_model}'

    if hasattr(args, 'cuda_cmp') and args.cuda_cmp:
            name += f'_cuda_cmp_{args.cuda_cmp}'
    return name


def visual_data(imgs,name):
    imgs_ =[]
    if isinstance(imgs[0],torch.Tensor):
        for i in range(len(imgs)):
            imgs_.append(imgs[i].numpy())
        imgs = imgs_
    fig,ax = plt.subplots(4,4,figsize=(12, 12))
    for i in range(4):
        for j in range(4):
            if not (isinstance(imgs[0], np.ndarray) and imgs[0].shape == (32, 32, 3)):
                ax[i,j].imshow(imgs[i*4+j].transpose(1, 2, 0))
            else:
                ax[i,j].imshow(imgs[i*4+j])
            ax[i,j].axis('off')
    plt.savefig(name)


def save_data(gradient,pos_num):
    conv = [[]for _ in range(pos_num)]
    conv_w = [[]for _ in range(pos_num)]
    # for grad in grads:
    #      gradient.extend(grad)

    for i in tqdm(range(0,len(gradient),pos_num)):
        for j in range(pos_num):
            conv[j].append(torch.mean(abs(gradient[i+j][1].clone())))
            conv_w[j].append(torch.mean(gradient[i+j][0].clone()))
        grad =[conv,conv_w]
    np.save(subdir+'gradinfo.npy',grad)


def plot_data_distribution(errdata,oridata,dcmpdata,name):
    plt.hist(oridata, bins=200, alpha=0.3, color='g',label=f'Original Data {len(oridata)}')#,density=True)
    plt.hist(dcmpdata, bins=200, alpha=0.3, color='r',label=f'Decompressed Data {len(dcmpdata)}')#,density=True)
    plt.hist(errdata, bins=200, alpha=0.8, color='b',label=f'Error Data {len(errdata)}')#,density=True)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(name+'.png')


def save_weights_hook(grad,param,pos,wandb):
    global weight_dict,weight_val_dict
    weight_mean = torch.mean(abs(param.data.clone()))
    weight = param.data.clone()
    gradients = torch.mean(abs(grad.clone()))
    if pos not in weight_dict:
        weight_dict[pos] = weight_mean.item()
        weight_val_dict[pos] = weight
    else:
        weightdiff_ratio = (weight_mean - weight_dict[pos])/weight_dict[pos]
        weight_dis = torch.norm(weight - weight_val_dict[pos])
        weight_dict[pos]=weight_mean.item()
        # weight_val_dict[pos] = weight
        wandb.log({f'position {pos} weight difference ratio':weightdiff_ratio,f'position {pos} gradient':gradients,f'position {pos} weight distance':weight_dis})


def getparam(model,pos):
    param3 = None
    if pos == 0:
        param1 = list(model.parameters())
        param2 = None
    if pos == 1:
        param1 = list(model.layer1[0].conv1.parameters())
        param2 = list(model.layer1[0].conv2.parameters())
    if pos == 2:
        param1 = list(model.layer1[1].conv1.parameters())
        param2 = list(model.layer1[1].conv2.parameters())          
    if pos == 3:
        param1 = list(model.layer1[2].conv1.parameters())
        param2 = list(model.layer1[2].conv2.parameters())
    if pos == 4:
        param1 = list(model.layer1[3].conv1.parameters())
        param2 = list(model.layer1[3].conv2.parameters())
    if pos == 5:
        param1 = list(model.layer1[4].conv1.parameters())
        param2 = list(model.layer1[4].conv2.parameters())
    if pos == 6:
        param1 = list(model.layer2[0].conv1.parameters())
        param2 = list(model.layer2[0].conv2.parameters())
        param3 = list(model.layer2[0].downsample[0].parameters())
    if pos == 7:
        param1 = list(model.layer2[1].conv1.parameters())
        param2 = list(model.layer2[1].conv2.parameters())
    if pos == 8:
        param1 = list(model.layer2[2].conv1.parameters())
        param2 = list(model.layer2[2].conv2.parameters())
    if pos == 9:
        param1 = list(model.layer2[3].conv1.parameters())
        param2 = list(model.layer2[3].conv2.parameters())
    if pos == 10:
        param1 = list(model.layer2[4].conv1.parameters())
        param2 = list(model.layer2[4].conv2.parameters())
    if pos == 11:
        param1 = list(model.layer3[0].conv1.parameters())
        param2 = list(model.layer3[0].conv2.parameters())
        param3 = list(model.layer3[0].downsample[0].parameters())
    if pos == 12:
        param1 = list(model.layer3[1].conv1.parameters())
        param2 = list(model.layer3[1].conv2.parameters())
    if pos == 13:
        param1 = list(model.layer3[2].conv1.parameters())
        param2 = list(model.layer3[2].conv2.parameters())
    if pos == 14:
        param1 = list(model.layer3[3].conv1.parameters())
        param2 = list(model.layer3[3].conv2.parameters())
    if pos == 15:
        param1 = list(model.layer3[4].conv1.parameters())
        param2 = list(model.layer3[4].conv2.parameters())
    
    if param3 is not None:
        return param1,param2,param3
    else:
        return param1,param2,None
    

def getparam_block(model,pos):
    if pos == 0:
        param = list(model.parameters())
    if pos == 1:
        param = list(model.layer1[0].parameters())
    if pos == 2:
        param = list(model.layer1[1].parameters())
    if pos == 3:
        param = list(model.layer1[2].parameters())
    if pos == 4:
        param = list(model.layer1[3].parameters())
    if pos == 5:
        param = list(model.layer1[4].parameters())
    if pos == 6:
        param = list(model.layer2[0].parameters())
    if pos == 7:
        param = list(model.layer2[1].parameters())
    if pos == 8:
        param = list(model.layer2[2].parameters())
    if pos == 9:
        param = list(model.layer2[3].parameters())
    if pos == 10:
        param = list(model.layer2[4].parameters())
    if pos == 11:
        param = list(model.layer3[0].parameters())
    if pos == 12:
        param = list(model.layer3[1].parameters())
    if pos == 13:
        param = list(model.layer3[2].parameters())
    if pos == 14:
        param = list(model.layer3[3].parameters())
    if pos == 15:
        param = list(model.layer3[4].parameters())

    return param


def get_handle(model,pos):
    if pos == 0:
        hook_handle = model.conv1
    elif pos == 1:
        hook_handle = model.layer1[0].conv2
    elif pos == 2:
        hook_handle = model.layer1[1].conv2
    elif pos == 3:
        hook_handle = model.layer1[2].conv2
    elif pos == 4:
        hook_handle = model.layer1[3].conv2
    elif pos == 5:
        hook_handle = model.layer1[4].conv2
    elif pos == 6:
        hook_handle = model.layer2[0].conv2
    elif pos == 7:
        hook_handle = model.layer2[1].conv2
    elif pos == 8:
        hook_handle = model.layer2[2].conv2
    elif pos == 9:
        hook_handle = model.layer2[3].conv2
    elif pos == 10:
        hook_handle = model.layer2[4].conv2
    elif pos == 11:
        hook_handle = model.layer3[0].conv2
    elif pos == 12:
        hook_handle = model.layer3[1].conv2
    elif pos == 13:
        hook_handle = model.layer3[2].conv2
    elif pos == 14:
        hook_handle = model.layer3[3].conv2
    elif pos == 15:
        hook_handle = model.layer3[4].conv2
    return hook_handle


def get_handle_front(model,pos):
    if pos == 0:
        hook_handle = model.bn
    elif pos == 1:
        hook_handle = model.layer1[0]
    elif pos == 2:
        hook_handle = model.layer1[1]
    elif pos == 3:
        hook_handle = model.layer1[2]
    elif pos == 4:
        hook_handle = model.layer1[3]
    elif pos == 5:
        hook_handle = model.layer1[4]
    elif pos == 6:
        hook_handle = model.layer2[0]
    elif pos == 7:
        hook_handle = model.layer2[1]
    elif pos == 8:
        hook_handle = model.layer2[2]
    elif pos == 9:
        hook_handle = model.layer2[3]
    elif pos == 10:
        hook_handle = model.layer2[4]
    elif pos == 11:
        hook_handle = model.layer3[0]
    elif pos == 12:
        hook_handle = model.layer3[1]
    elif pos == 13:
        hook_handle = model.layer3[2]
    elif pos == 14:
        hook_handle = model.layer3[3]
    elif pos == 15:
        hook_handle = model.layer3[4]
    return hook_handle


def get_handle_front_res50(model,pos):
    if pos == 0:
        hook_handle = model.bn1
    elif pos == 1:
        hook_handle = model.layer1[0]
    elif pos == 2:
        hook_handle = model.layer1[1]
    elif pos == 3:
        hook_handle = model.layer1[2]
    elif pos == 4:
        hook_handle = model.layer2[3]
    elif pos == 5:
        hook_handle = model.layer2[0]
    elif pos == 6:
        hook_handle = model.layer2[1]
    elif pos == 7:
        hook_handle = model.layer2[2]
    elif pos == 8:
        hook_handle = model.layer2[3]
    elif pos == 9:
        hook_handle = model.layer3[0]
    elif pos == 10:
        hook_handle = model.layer3[1]
    elif pos == 11:
        hook_handle = model.layer3[2]
    elif pos == 12:
        hook_handle = model.layer3[3]
    elif pos == 13:
        hook_handle = model.layer3[4]
    elif pos == 14:
        hook_handle = model.layer3[5]
    elif pos == 15:
        hook_handle = model.layer4[0]
    elif pos == 16:
        hook_handle = model.layer4[1]
    elif pos == 17:
        hook_handle = model.layer4[2]
    return hook_handle


def getparam_block_res50(model,pos):
    if pos == 0:
        param = list(model.conv1.parameters())+list(model.bn1.parameters())
    if pos == 1:
        param = list(model.layer1.parameters())
    if pos == 2:
        param = list(model.layer2.parameters())
    if pos == 3:
        param = list(model.layer3.parameters())
    if pos == 4:
        param = list(model.layer4.parameters())

    return param


def check_freez_layer(model,posi):
    for pos in range(posi+1):
        if pos > 0:
            params1,params2,params3 = getparam(model,pos)
            is_frozen1 = all(not param.requires_grad for param in params1)
            is_frozen2 = all(not param.requires_grad for param in params2)
            if params3 is not None:
                is_frozen3 = all(not param.requires_grad for param in params3)
                if not is_frozen3:
                    for param3 in params3:
                        param3.requires_grad = False
                        param3.grad = None
            if not (is_frozen1 and is_frozen2):
                for param1 in params1:
                    param1.requires_grad = False
                    param1.grad = None
                for param2 in params2:
                    param2.requires_grad = False
                    param2.grad = None
            print(f'position {pos} has been freezed')    
        if pos == 0:
            params1,_,_ = getparam(model,pos)
            is_frozen1 = all(not param.requires_grad for param in params1)
            if not is_frozen1:
                for param1 in params1:
                    param1.requires_grad = False
                    param1.grad = None
            print(f'position {pos} has been freezed')
    # for pos in range(posi+1,16):
    #     if pos > 0:
    #         params1,params2,params3 = getparam(model,pos)
    #         is_active1 = all( param.requires_grad for param in params1)
    #         is_active2 = all( param.requires_grad for param in params2)
           
    #         if params3 is not None:
    #             is_active3 = all( param.requires_grad for param in params3)
    #             if not (is_active3):
    #                     for param3 in params3:
    #                         param3.requires_grad = True
    #                 # param3.grad = None
    #         if not (is_active1 and is_active2):
    #             for param1 in params1:
    #                 param1.requires_grad = True
    #                 #  param1.grad = None
    #             for param2 in params2:
    #                 param2.requires_grad = True
    #                 #  param2.grad = None
            

    return 

    # if pos < 0:
    #     return


def check_freez_block(model,posi):
    for pos in range(posi+1):
        params= getparam_block(model,pos)
        is_frozen = all(not param.requires_grad for param in params)
        if not is_frozen:
            for param in params:
                 param.requires_grad = False
                 param.grad = None
        print(f'position {pos} has been freezed') 


def check_active_block(model,posi):
    for pos in range(posi,16):
        params= getparam_block(model,pos)
        is_frozen = all(param.requires_grad for param in params)
        if not is_frozen:
            for param in params:
                 param.requires_grad = True
                #  param.grad = None
            print(f'position {pos} has been unfreezed') 
    param_lin = model.linear.parameters()
    is_frozen = all(param.requires_grad for param in param_lin)
    if not is_frozen:
        for param in param_lin:
            param.requires_grad = True
        print(f'position {16} has been unfreezed')  
    return
    
  
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
            decompressed_array_cal = decompressed_array.copy()
            t4 = time.time()
            act_vle = np.mean(np.abs(output))
            noise = decompressed_array_cal - output
            # noise_mean = np.mean(np.abs(noise))
            # noise_ratio = noise_mean/act_vle
            # cos_sim = 1 - cosine(output.flatten(), decompressed_array_cal.flatten())
            wandb.log({f'cmp_ratio':intersize_cmpd/intersize,f'error ratio':noise_ratio,f'cosine similarity':cos_sim})
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


class AvgPoolAndFlatten(nn.Module):
    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[3])  # 全局池化
        x = x.view(x.size(0), -1)         # 展平
        return x


def seperate_model(model,pos):
    def flatten_sequential(modules):
        flattened = []
        for module in modules:
            if isinstance(module, nn.Sequential):
                flattened.extend(module.children())
            else:
                flattened.append(module)
        return flattened
    def insert_relu_after_second_layer(sequential_model):
        layers = list(sequential_model.children())  # 将原始模型的层转为列表
        if len(layers) >= 2:
            layers.insert(2, nn.ReLU())  # 在第二层之后插入 ReLU
        return nn.Sequential(*layers)
     # 初始化键名映射
    mapping = {}
    if pos == 0:
        front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:2]))
        remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[2:]))
    elif pos == 16:
        front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:-1]))
        remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[-1:]))
    elif pos>=1 and pos<=15:
        layer_num=(pos-1)//5
        block_num=(pos-1)%5
        if block_num == 4:
            front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:2 + layer_num+1]))
            remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[2 + layer_num + 1:]))
        front_layers = nn.Sequential(
                *copy.deepcopy(flatten_sequential(list(model.children())[:2 + layer_num]) + 
                    flatten_sequential(list(model.children())[2 + layer_num ][:block_num + 1]))
                )
        remaining_layers = nn.Sequential(
                *copy.deepcopy(flatten_sequential(list(model.children())[2 + layer_num][block_num + 1:])
                + flatten_sequential(list(model.children())[2 + layer_num + 1:])))
        assert len(front_layers)+len(remaining_layers)==18, "seprarate model lack some part"

    front_layers = insert_relu_after_second_layer(front_layers)  # 在前半部分插入 ReLU

    # 添加键名映射
    original_state_dict = model.state_dict()
    front_state_dict = front_layers.state_dict()
    

    mapping_front = {}  # 拆分后键名 → 带前缀键名
    mapping_remain = {}  # 带前缀键名 → 原始模型键名
    mapping_final = {}  # 最终映射：拆分后键名 → 原始模型键名
    used_original_keys = set()

    # 为前半部分添加映射
    for key in front_state_dict.keys():
        prefixed_key = f"f.{key}"  # 带前缀的键名
        mapping_front[key] = prefixed_key
        for original_key in original_state_dict.keys():
            # 确保匹配 shape 且原始键未被使用
            if ( front_state_dict[key].shape == original_state_dict[original_key].shape
                and original_key not in used_original_keys
            ):
                mapping_final[prefixed_key] = original_key
                used_original_keys.add(original_key)  # 标记为已使用
                break

    

    # add the average pooling and flatten layer
    if len(remaining_layers) > 1:
        new_remaining_layers = nn.Sequential(
            *list(remaining_layers.children())[:-1],  # 剔除最后一层
            AvgPoolAndFlatten(),                      # 添加自定义模块
            list(remaining_layers.children())[-1]     # 添加最后一层
        )
    else:
        new_remaining_layers = nn.Sequential(
            AvgPoolAndFlatten(),
            *list(remaining_layers.children())
        )
        # 为后半部分添加映射

    remain_state_dict = new_remaining_layers.state_dict()
    for key in remain_state_dict.keys():
        prefixed_key = f"r.{key}"  # 带前缀的键名
        mapping_remain[key] = prefixed_key
        for original_key in original_state_dict.keys():
            if ( remain_state_dict[key].shape == original_state_dict[original_key].shape
                and original_key not in used_original_keys
            ):
                mapping_final[prefixed_key] = original_key
                used_original_keys.add(original_key)  # 标记为已使用
                break

    mapping_list=[mapping_front,mapping_remain,mapping_final]

    return front_layers,new_remaining_layers,mapping_list


def merge_models(front_model, remain_model, new_model, mappinglist):
    front_mapping, remain_mapping, final_mapping = mappinglist
    
    new_state_dict = new_model.state_dict()
    
    # 合并前半部分
    front_state_dict = front_model.state_dict()
    for split_key, prefixed_key in front_mapping.items():
        # 通过 final_mapping 找到原始模型键名
        if prefixed_key in final_mapping:
            original_key = final_mapping[prefixed_key]
            if original_key in new_state_dict:
                # 将前半部分模型的参数赋值到新模型
                new_state_dict[original_key] = front_state_dict[split_key]

    # 加载后半部分参数
    remain_state_dict = remain_model.state_dict()
    for split_key, prefixed_key in remain_mapping.items():
        # 通过 final_mapping 找到原始模型键名
        if prefixed_key in final_mapping:
            original_key = final_mapping[prefixed_key]
            if original_key in new_state_dict:
                # 将后半部分模型的参数赋值到新模型
                new_state_dict[original_key] = remain_state_dict[split_key]


    # 加载合并后的参数
    new_model.load_state_dict(new_state_dict, strict=True)
 
    return new_model


def shuffle_data(data,):
    indices = torch.randperm(data.size(0))
    data_shuffled = data[indices]
    return data_shuffled


def calculate_model_memory(model, input_size, device='cuda'):
    model = model.to(device)
    dtype_size = torch.tensor([], dtype=torch.float32).element_size()  # 单个float32占用的字节数
    
    # 计算权重和权重梯度的内存占用
    weight_memory = sum(p.numel() for p in model.parameters()) * dtype_size
    grad_memory = sum(p.numel() for p in model.parameters() if p.requires_grad) * dtype_size
    
    # Hook 用于记录激活值和激活值梯度
    activation_memory = [0]  # 使用列表方便闭包修改
    gradient_memory = [0]

    def forward_hook(module, input, output):
        if isinstance(output, tuple):  # 兼容多输出的情况
            for o in output:
                activation_memory[0] += o.numel() * dtype_size
        else:
            activation_memory[0] += output.numel() * dtype_size

    def backward_hook(module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            for go in grad_output:
                if go is not None:
                    gradient_memory[0] += go.numel() * dtype_size
        else:
            gradient_memory[0] += grad_output.numel() * dtype_size
    
    # 注册 Hook
    hooks = []
    for module in model.modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(forward_hook))
            hooks.append(module.register_backward_hook(backward_hook))
    
    # 模拟一次前向和后向传播
    dummy_input = torch.randn(*input_size, device=device)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    
    # 移除 Hook
    for hook in hooks:
        hook.remove()
    
    total_memory = weight_memory + grad_memory + activation_memory[0] + gradient_memory[0]
    return {
        "weight_memory": weight_memory/1024**2,
        "gradient_memory": grad_memory/1024**2,
        "activation_memory": activation_memory[0]/1024**2,
        "activation_gradient_memory": gradient_memory[0]/1024**2,
        "total_memory": total_memory/1024**2
    }

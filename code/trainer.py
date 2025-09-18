import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
# import resnet
from cad_utils import get_handle_front,save_weights_hook,shuffle_data

import zfpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_names = sorted(name for name in resnet.__dict__
#     if name.islower() and not name.startswith("__")
#                      and name.startswith("resnet")
#                      and callable(resnet.__dict__[name]))
mask = None
def trans_mask():
    global mask
    return mask



# print(model_names)
# run = neptune.init_run(
#     project="ycc-wandb/Resnet20-accuarcy",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0NTgzYS1iNDFhLTQwZDEtODAyOS1iYzU1OGQzMWEwODQifQ==",
# )
# params = {"learning_rate": 0.1, "optimizer": "SGD momentum_0.9"}
# run["parameters"] = params

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
#                     choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) +
#                     ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))


def train(train_loader, model, criterion, optimizer, epoch, args, wandb = None, logger = None):
    global mask
    dcmp_time =None
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # run = neptune.init_run(
    # project="ycc-wandb/Resnet20-accuarcy",
    # api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0NTgzYS1iNDFhLTQwZDEtODAyOS1iYzU1OGQzMWEwODQifQ==",)

    # switch to train mode
    model.train()

    end = time.time()
    t_data_start = time.perf_counter()
    for i, batch in enumerate(train_loader):

        
        
        if len(batch) == 4:
            augmented_batch, original_batch, labels, mask = batch
            input = torch.cat((augmented_batch, original_batch), dim=0)  # 在第0维（batch 维度）拼接
            target = torch.cat((labels, labels), dim=0) # 拼接后的标签，重复原始标签
        elif len(batch) == 3:
            # augmented_batch, original_batch, labels = batch
            # input = torch.cat((augmented_batch, original_batch), dim=0)  # 在第0维（batch 维度）拼接
            # target = torch.cat((labels, labels), dim=0) # 拼接后的标签，重复原始标签
            input, target, dcmp_time = batch
            if len(input.size()) != 4:
                input = input.reshape(input.size(0)*input.size(1),input.size(2),input.size(3),input.size(4))
                target = target.reshape(target.size(0)*target.size(1))
                indices = torch.randperm(input.size(0))
                input = input[indices]
                target = target[indices]
        elif len(batch) == 2:
            input, target = batch
            if len(input.size()) != 4:
                input = input.reshape(input.size(0)*input.size(1),input.size(2),input.size(3),input.size(4))
                target = target.reshape(target.size(0)*target.size(1))
                indices = torch.randperm(input.size(0))
                input = input[indices]
                target = target[indices]
        
        # if len(input) != args.batch_size:å

        # measure data loading time
        t_data_end = time.perf_counter()
        data_time.update(time.time() - end)
        dcmp_time_avg = None
        
        if dcmp_time is not None and len(dcmp_time) > 0:
            dcmp_time_avg = torch.mean(dcmp_time)
            dcmp_time = []


        

        t0 = time.perf_counter()
        target = target.to(device,non_blocking=True)
        target_var = target
        if target_var.dim() > 1 and target_var.size(1) == 1:
                 target_var = target_var.squeeze(1)  # 仅在有第 1 维且大小为 1 时使用
       
        
        # if dropstatus:
        #     input_dcmp = zfpy.decompress_numpy(input)
        #     input_dec = torch.from_numpy(input_dcmp).to('cuda')
        #     input_var = input_dec.cuda()
        # else:
        input_var = input.to(device,non_blocking=True)
        t1 = time.perf_counter()
        # if i == 781:
        #     input_var=input_var


        # compute output
        output = model(input_var)
        
        loss = criterion(output, target_var)
        t2 = time.perf_counter()

        
        if not output.requires_grad:
            output.requires_grad = True
        if not loss.requires_grad:
            loss.requires_grad = True
        # run["train/loss"].append(loss)
        # loss.requires_grad = True

        


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t3 = time.perf_counter()

        trans_t = t1 - t0
        logger.set_transfer(trans_t)
        forward_t = t2 - t1
        logger.set_forward(forward_t)
        backward_t = t3 - t2
        logger.set_backward(backward_t)
        load_data_t = t_data_end - t_data_start
        logger.set_load_time(load_data_t)

        logger.commit_iter_row(i)

        # if hasattr(args, 'freez_epoch') and args.freez_epoch:
        if hasattr(args, 'forward_hook') and args.forward_hook:
            # if epoch >= args.freez_epoch[-1]:
                for forward_flag in range(16):
                    fwd_handle = get_handle_front(model,forward_flag)
                    if forward_flag == 0: # the first conv layer has no children modules
                        if fwd_handle.weight.grad is not None:
                            grad = fwd_handle.weight.grad
                        else:
                            grad = torch.zeros_like(fwd_handle.weight)
                            save_weights_hook(grad, fwd_handle.weight, f'{forward_flag}_conv',wandb)

                    if fwd_handle is not None:
                            # 遍历该 block 内的所有子模块
                        for name, layer in fwd_handle.named_children():
                            # 检查每个子层是否有 weight 属性
                            if hasattr(layer, 'weight'):
                                if layer.weight.grad is not None:
                                    grad = layer.weight.grad
                                else:
                                    grad = torch.zeros_like(layer.weight)
                                
                                # 保存每个子层的权重和梯度
                                save_weights_hook(grad, layer.weight, f'{forward_flag}_{name}',wandb)

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        if wandb is not None:
            wandb.log({"loss": loss,"train_acc":prec1})

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if logger is not None:
            logger.reset_iter()



        if i % 50 == 0:#args.print_freq == 0:
            if dcmp_time_avg is not None:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})({dcmp_time_avg:.5f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time,dcmp_time_avg=dcmp_time_avg, loss=losses, top1=top1))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1))
        t_data_start = time.perf_counter()

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()
            if target_var.dim() > 1 and target_var.size(1) == 1:
                 target_var = target_var.squeeze(1)  # 仅在有第 1 维且大小为 1 时使用
     

            # if args.half:
                # input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0: #args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

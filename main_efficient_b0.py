""" 
This module is for training another block after pretraining another block before
5c was trained before 5b is trained right now
"""
import argparse
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
from efficientnet_pytorch.model_feedback_v2_b0 import EfficientNet_F as EfficientNet_F2_B0
import PIL
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import optim.rmsprop as rmsprop 
import optim
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data',default="/dataset-imagenet-ilsvrc2012/",
                    help='path to dataset')

parser.add_argument('--model',default="resnet_feedback",
                    help='which network')
parser.add_argument('--agr', type = bool,default=False)


parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=True,
                    help='use pre-trained model')
parser.add_argument('--num_loops', type=int,default=2)
parser.add_argument('--pretrained_path', default= "",
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpus', type=str, default="0,1,2,3",#,4,5,6,7",
                    help="define gpu ids")

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
   # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    main_worker(args.gpus, args)


def main_worker(gpu,  args):
    global best_acc1
    print("You are using:",torch.cuda.device_count()," gpus")
    model = EfficientNet_F2_B0.build_model('efficientnet-b0') 


    print("number of trainable parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    dict = torch.load(args.pretrained_path)#['state_dict']
    args.pretrained_path
    if args.pretrained and not args.evaluate:
       print("Loading:",args.pretrained_path)
#       model.load_state_dict(dict,strict=True)    
       model.load_state_dict(dict,strict=False)    
    if args.pretrained and args.evaluate:
       model.load_state_dict(dict,strict=True)    


    model = model.cuda()
    model = nn.DataParallel(model) 
    # create model
    

        

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    parameters= []
    
    ft_module_names = ['module.convhead_create_dy_w.weight','module.convhead_create_dy_w.bias']#,'module._fc.weight','module._fc.bias'] 
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                print("Parameter:",k," is getting different lr")
                parameters.append({'params': v, 'lr': 0.0125})
                break
        else:
             parameters.append({'params': v,'lr':args.lr})
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,dampening=0,
                                weight_decay=args.weight_decay)

    #optimizer = rmsprop.TFRMSprop(parameters, lr=0.000125, rho=0.9, eps=1e-03, weight_decay=0, momentum=0.9, warmup=200185)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98738885893)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = 79.04
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            #for group in optimizer.param_groups:
            #   group['dampening '] = 0

   
            #t(optimizer)
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join("/dataset-imagenet-ilsvrc2012/", 'train')
    valdir = os.path.join("/dataset-imagenet-ilsvrc2012/", 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=PIL.Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.RandomErasing(),
            normalize,
        ]))
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        #scheduler.step()
        train(train_loader, model, criterion, optimizer, epoch, args)
        #scheduler.step()
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    torch.autograd.set_detect_anomaly(True)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()


    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        images = images.cuda()
        target = target.cuda()
        lossarr = []
        # compute output
        outputs = model(images)
        if isinstance(outputs, list):
           if len(outputs) > 1:
              output = outputs[-1]
              for out in outputs:
                 lossarr.append(criterion(out, target))
              first_diff = torch.max(lossarr[-1]-lossarr[-2], torch.zeros(lossarr[-1].shape, dtype=torch.float32).cuda())
              if len(lossarr) == 2:
                  loss = lossarr[-1] + 0.1*first_diff + 0.001*lossarr[-2]
              elif len(lossarr) == 3:
                 sec_diff = torch.max(lossarr[-2]-lossarr[-3], torch.zeros(lossarr[-1].shape, dtype=torch.float32).cuda())
                 loss = lossarr[-1] + 0.1*first_diff + 0.001*lossarr[-2] + 0.1*sec_diff + 0.001*lossarr[-3] 
              elif len(lossarr) == 4:
                 sec_diff = torch.max(lossarr[-2]-lossarr[-3], torch.zeros(lossarr[-1].shape, dtype=torch.float32).cuda())
                 third_diff = torch.max(lossarr[-3]-lossarr[-4], torch.zeros(lossarr[-1].shape, dtype=torch.float32).cuda())
                 loss = lossarr[-1] + 0.1*first_diff + 0.002*lossarr[-2] + 0.1*sec_diff + 0.001*lossarr[-3] + 0.1*third_diff + 0.004*lossarr[-4]#op2
           else:
             output = outputs[-1]
             loss = criterion(output, target) 
        else:     
             loss = criterion(outputs, target)            
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    num_gpus = torch.cuda.device_count()
   
    

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpus is not None:
                images = images.cuda()#args.gpu, non_blocking=True)
            target = target.cuda()#args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            if isinstance(output, list):
               output = output[-1]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
     torch.save(state, "efficentnetb0_0.0125_0.00125_checkpoint.pth")
     if is_best:
        print("Best, Saving")
        torch.save(state, "efficentnetb0_0.0125_0.00125_best.pth")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
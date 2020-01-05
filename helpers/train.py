from . experiment_file_management import save_trial_results, load_trial_results


class TrainingManager():
    def __init__(self, trial_name, model, load):
        self.trial_name          = trial_name
        self.iter_no             = 0,
        self.train_loss          = [],
        self.train_acc           = [],
        self.test_loss           = [],
        self.test_acc            = [],
        self.model_state_dict    = None

        if load:
            self._load()
        
    def _load(self):
        print('Loading previous TrainingManager')
        for k, v in load_trial_results(self.trial_name).items():
            setattr(self, k, v)

    def _get_recent_view(self, l, mean_range):
        return l if len(l) <= mean_range else l[-mean_range:]

    def _get_recent_mean(self, l, mean_range):
        import numpy as np
        return np.mean(self._get_recent_view(l, mean_range))
 
    def _repr(self, mean_range):
        from functools import partial
        return '{tr_loss: %.5f, tr_acc: %.5f, te_loss: %.5f, te_acc: %.5f}' % \
                tuple(map(partial(self._get_recent_mean, mean_range=mean_range), 
                [self.train_loss, self.train_acc, self.test_loss, self.test_acc]))

    def _save(self, is_trial):
        from collections import OrderedDict
        self.model_state_dict = OrderedDict({k:v.to('cpu') for k, v in self.model.state_dict().items()})
        if is_trial:
            save_trial_results(self.trial_name, self.__dict__)

    def _add(self, tr_loss, tr_acc, te_loss, te_acc):
        self.iter_no += 1
        for v,l in zip([tr_loss, tr_acc, te_loss, te_acc], [self.train_loss, self.train_acc, self.test_loss, self.test_acc]):
            l.append(float(v))

    def train(self, model, optimizer, trainloader, testloader, 
                    criterion_train, criterion_test, acc_train, acc_test,
                    test_iter_each_x_train_iter=5, mean_range=100, save_rate=1000,
                    desc_update_rate=5, no_iterations=50000):
        from tqdm.auto import tqdm
        import time
        last_save_time = last_update_time = 0
        
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict)
        
        model.train()
        
        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x
        
        trainiter, testiter = [iter(cycle(trainloader)), iter(cycle(testloader))]
        get_next = lambda it: map(lambda x: x.to(device), next(it))
        
        first=True
        with tqdm(range(self.iter_no, no_iterations), initial=self.iter_no, total=no_iterations) as tq:
            for i in tq:
                optimizer.zero_grad()
                inputs, labels = get_next(trainiter)
                outputs = self.model(inputs)
                tr_loss = criterion_train(outputs, labels)
                tr_loss.backward()
                optimizer.step()
                tr_acc = acc_train(outputs, labels)
                
                if i % test_iter_each_x_train_iter == 0 or first:
                    first = False
                    model.eval()
                    with torch.no_grad():
                        inputs, labels = get_next(testiter)
                        outputs = self.model(inputs)
                        te_loss = criterion_test(outputs, labels)
                        te_acc = acc_test(outputs, labels)
                        model.train()
                
                self._add(tr_loss, tr_acc, te_loss, te_acc)
                if time.time() - last_update_time > desc_update_rate:
                    last_update_time = time.time()
                    tq.set_description(self._repr())
                if time.time() - last_save_time > save_rate:
                    last_save_time = time.time()
                    self._save()
                
        self._save()

def test():
    from tqdm.auto import tqdm
    for i in tqdm(range(100000000)):
        x = i**2
 
#  if args.model == "resnet20_feedback":
#  model = ResNet20dy(args.original_weights,args.num_loops)
#  elif args.model == "resnet20":
#  model = resnet.__dict__[args.arch]()
#  model = model.cuda()
#  model = torch.nn.DataParallel(model)
#  if args.pretrained:
#  try:
#  dict=torch.load(args.pretrained_path)['state_dict']
#  except:
#  dict=torch.load(args.pretrained_path)
#  if "feedback" in args.model:
#  new_dict = {}
#  for k,v in dict.items():
#  if "layer3.2" in k:
#  k = k.replace("layer3.2","layer3.block3") 
#  elif "layer3.1" in k:
#  k = k.replace("layer3.1","layer3.block2")
#  elif "layer3.0" in k:
#  k = k.replace("layer3.0","layer3.block1")
#  new_dict[k] = v
#  else:
#  new_dict = dict 

#  model.load_state_dict(new_dict,strict=False)

#  # optionally resume from a checkpoint
#  if args.resume:
#  if os.path.isfile(args.resume):
#  print("=> loading checkpoint '{}'".format(args.resume))
#  checkpoint = torch.load(args.resume)
#  args.start_epoch = checkpoint['epoch']
#  best_prec1 = checkpoint['best_prec1']
#  model.load_state_dict(checkpoint['state_dict'])
#  print("=> loaded checkpoint '{}' (epoch {})"
#  .format(args.evaluate, checkpoint['epoch']))
#  else:
#  print("=> no checkpoint found at '{}'".format(args.resume))

#  cudnn.benchmark = True

#  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#  std=[0.229, 0.224, 0.225])

#  train_loader = torch.utils.data.DataLoader(
#  datasets.CIFAR10(root=args.dataset_path, train=True, transform=transforms.Compose([
#  transforms.RandomHorizontalFlip(),
#  transforms.RandomCrop(32, 4),
#  transforms.ToTensor(),
#  transforms.RandomErasing(),
#  normalize,
#  ]), download=True),
#  batch_size=args.batch_size, shuffle=True,
#  num_workers=args.workers, pin_memory=True)

#  val_loader = torch.utils.data.DataLoader(
#  datasets.CIFAR10(root=args.dataset_path, train=False, transform=transforms.Compose([
#  transforms.ToTensor(),
#  normalize,
#  ])),
#  batch_size=128, shuffle=False,
#  num_workers=args.workers, pin_memory=True)

 
#  criterion = nn.CrossEntropyLoss().cuda()

#  parameters= []
#  ft_module_names = ['layer3.block3.block3_conv1_create_dy_w.weight','layer3.block3.block3_conv1_create_dy_w.bias']
#  for k, v in model.named_parameters():
#  for ft_module in ft_module_names:
#  if ft_module in k:
#  print(k,"is getting higher lr")
#  parameters.append({'params': v, 'lr': 2})
#  break
#  else:
#  parameters.append({'params': v})

#  optimizer = torch.optim.SGD(parameters, args.lr,
#  momentum=args.momentum,
#  weight_decay=args.weight_decay) 
#  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#  milestones=[50, 100,150], last_epoch=args.start_epoch - 1)

#  if args.evaluate:
#  validate(val_loader, model, criterion)
#  return

#  tb = SummaryWriter(args.tb_filename)
#  for epoch in range(args.start_epoch, args.epochs):
#  print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
#  train_top1 = train(train_loader, model, criterion, optimizer, epoch)
#  lr_scheduler.step() 
#  tb.add_scalar('Train_Top1',train_top1,epoch) 
#  prec1 = validate(val_loader, model, criterion)
#  tb.add_scalar('Val_Top1',prec1,epoch) 
#  # remember best prec@1 and save checkpoint
#  is_best = prec1 > best_prec1
#  best_prec1 = max(prec1, best_prec1)
#  if epoch > 0 and is_best:
#  name = os.path.join(args.save_path,'ep' + str(epoch) + '_acc' + str(prec1) + ".pth")
 
#  print("This is best the weight so far with acc = prec1, saving with name:",name)
#  save_checkpoint({
#  'epoch': epoch + 1,
#  'state_dict': model.state_dict(),
#  'best_prec1': best_prec1,
#  }, is_best, filename=name)
#  """
#  save_checkpoint({
#  'state_dict': model.state_dict(),
#  'best_prec1': best_prec1,
#  }, is_best, filename=args.save_path)
#  """


# def train(train_loader, model, criterion, optimizer, epoch):
#  """
#  Run one train epoch
#  """
#  batch_time = AverageMeter()
#  data_time = AverageMeter()
#  losses = AverageMeter()
#  top1 = AverageMeter()

#  # switch to train mode
#  model.train()

#  end = time.time()
#  for i, (input, target) in enumerate(train_loader):

#  # measure data loading time
#  data_time.update(time.time() - end)

#  target = target.cuda()
#  input_var = input.cuda()
#  target_var = target
#  if args.half:
#  input_var = input_var.half()

#  # compute output
#  output = model(input_var,args.alpha)
#  loss = criterion(output, target_var)

#  # compute gradient and do SGD step
#  optimizer.zero_grad()
#  loss.backward()
#  optimizer.step()
#  output = output.float()
#  loss = loss.float()
 
#  # measure accuracy and record loss
#  prec1 = accuracy(output.data, target)[0]
#  losses.update(loss.item(), input.size(0))
#  top1.update(prec1.item(), input.size(0))

#  # measure elapsed time
#  batch_time.update(time.time() - end)
#  end = time.time()

#  if i % args.print_freq == 0:
#  print('Epoch: [{0}][{1}/{2}]\t'
#  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#  epoch, i, len(train_loader), batch_time=batch_time,
#  data_time=data_time, loss=losses, top1=top1))
#  return top1.avg

# def validate(val_loader, model, criterion):
#  """
#  Run evaluation
#  """
#  batch_time = AverageMeter()
#  losses = AverageMeter()
#  top1 = AverageMeter()

#  # switch to evaluate mode
#  model.eval()

#  end = time.time()
#  with torch.no_grad():
#  for i, (input, target) in enumerate(val_loader):
#  target = target.cuda()
#  input_var = input.cuda()
#  target_var = target.cuda()

#  if args.half:
#  input_var = input_var.half()

#  # compute output
#  output = model(input_var,args.alpha)
#  loss = criterion(output, target_var)

#  output = output.float()
#  loss = loss.float()

#  # measure accuracy and record loss
#  prec1 = accuracy(output.data, target)[0]
#  losses.update(loss.item(), input.size(0))
#  top1.update(prec1.item(), input.size(0))

#  # measure elapsed time
#  batch_time.update(time.time() - end)
#  end = time.time()

#  if i % args.print_freq == 0:
#  print('Test: [{0}/{1}]\t'
#  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#  i, len(val_loader), batch_time=batch_time, loss=losses,
#  top1=top1))
#  print(' * Prec@1 {top1.avg:.3f},* BestPrec@1 {best:.3f}'
#  .format(top1=top1,best = best_prec1))

#  return top1.avg

# def save_checkpoint(state, is_best, filename):
#  """
#  Save the training model
#  """
#  torch.save(state, filename)

# class AverageMeter(object):
#  """Computes and stores the average and current value"""
#  def __init__(self):
#  self.reset()

#  def reset(self):
#  self.val = 0
#  self.avg = 0
#  self.sum = 0
#  self.count = 0

#  def update(self, val, n=1):
#  self.val = val
#  self.sum += val * n
#  self.count += n
#  self.avg = self.sum / self.count


# def accuracy(output, target, topk=(1,)):
#  """Computes the precision@k for the specified values of k"""
#  maxk = max(topk)
#  batch_size = target.size(0)

#  _, pred = output.topk(maxk, 1, True, True)
#  pred = pred.t()
#  correct = pred.eq(target.view(1, -1).expand_as(pred))

#  res = []
#  for k in topk:
#  correct_k = correct[:k].view(-1).float().sum(0)
#  res.append(correct_k.mul_(100.0 / batch_size))
#  return res


# if __name__ == '__main__':
#  main()

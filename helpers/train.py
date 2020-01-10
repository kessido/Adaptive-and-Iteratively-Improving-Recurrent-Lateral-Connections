from . experiment_file_management import save_trial_results, load_trial_results, is_log_exist
from . statistics import mean_of_last_k
from . torch_utils import load_state_dict
import torch
from tqdm import tqdm
import time

class TrainingManager():
    def __init__(self, trial_name, load, is_trial):
        self.trial_name          = trial_name
        self.is_trial            = is_trial
        self.iter_no             = 0
        self.train_loss          = []
        self.train_acc           = []
        self.test_loss           = []
        self.test_acc            = []
        self.model_state_dict    = None

        if load:
            self._load()
            self.is_trial = is_trial
        elif is_trial:
            assert(is_log_exist(trial_name) == False)
    
    def train(self, model, optimizer, trainloader, testloader, 
                    criterion_train, criterion_test, acc_train, acc_test,
                    test_iter_each_x_train_iter=5, mean_range=100, save_rate=1000,
                    desc_update_rate=5, no_iterations=50000, device=torch.device('cpu')):
        from termcolor import colored
        print('Start training trial:', colored(self.trial_name, 'blue'), colored('is_trial', 'green' if self.is_trial else 'red'))
        if 'cuda' in device.type:
            torch.backends.cudnn.benchmark = True
        last_save_time = last_update_time = 0
        
        if self.model_state_dict is not None:
            print("Loading state from loaded checkpoint")
            load_state_dict(model, self.model_state_dict)
        model.train()
        
        trainiter, testiter = [iter(_cycle(trainloader)), iter(_cycle(testloader))]
        get_next = lambda it: map(lambda x: x.to(device), next(it))
        
        first=True
        with tqdm(range(self.iter_no, no_iterations), initial=self.iter_no, total=no_iterations) as tq:
            for i in tq:
                optimizer.zero_grad()
                tr_loss, tr_acc = self._iteration(model, criterion_train, acc_train, *get_next(trainiter))
                tr_loss.backward()
                optimizer.step()
                if i % test_iter_each_x_train_iter == 0 or first:
                    first = False
                    model.eval()
                    with torch.no_grad():
                        te_loss, te_acc = self._iteration(model, criterion_test, acc_test, *get_next(testiter))
                    model.train()
                
                self._add(tr_loss, tr_acc, te_loss, te_acc)
                if time.time() - last_update_time > desc_update_rate:
                    last_update_time = time.time()
                    tq.set_description(self._repr(mean_range))
                if time.time() - last_save_time > save_rate:
                    last_save_time = time.time()
                    self._save(model)

        self._save(model)

    def _load(self):
        print('Loading previous TrainingManager')
        for k, v in load_trial_results(self.trial_name).items():
            setattr(self, k, v)
        
    def _repr(self, mean_range):
        from functools import partial
        return '{tr_loss: %.5f, tr_acc: %.5f, te_loss: %.5f, te_acc: %.5f}' % \
                tuple(map(partial(mean_of_last_k, k=mean_range), 
                [self.train_loss, self.train_acc, self.test_loss, self.test_acc]))

    def _save(self, model):
        from collections import OrderedDict
        self.model_state_dict = OrderedDict({k:v.to('cpu') for k, v in model.state_dict().items()})
        if self.is_trial:
            save_trial_results(self.trial_name, self.__dict__)

    def _add(self, tr_loss, tr_acc, te_loss, te_acc):
        self.iter_no += 1
        for v,l in zip([tr_loss, tr_acc, te_loss, te_acc], [self.train_loss, self.train_acc, self.test_loss, self.test_acc]):
            l.append(float(v))            
    
    def _iteration(self, model, criterion_fn, acc_fn, inputs, labels):
        outputs = model(inputs)
        loss = criterion_fn(outputs, labels)
        acc = acc_fn(outputs, labels)
        return loss, acc

def _cycle(iterable):
    while True:
        for x in iterable:
            yield x

#  optimizer = torch.optim.SGD(parameters, args.lr,
#  momentum=args.momentum,
#  weight_decay=args.weight_decay) 
#  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#  milestones=[50, 100,150], last_epoch=args.start_epoch - 1)
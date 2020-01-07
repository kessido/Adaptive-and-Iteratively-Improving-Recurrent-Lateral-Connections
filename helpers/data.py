from functools import partial
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

def get_dataloaders(dataset_name, batch_size, num_workers=20):
    trainset, testset = _datasets_get_func[dataset_name]()
    dataloader = partial(torch.utils.data.DataLoader, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return dataloader(trainset, shuffle=True), dataloader(testset)
        
    
    batch_size, num_workers

_datasets_base_folder = '/media/data1/idokessler'
_datasets_get_func = {}

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_transform_test_cifar10 = transforms.Compose([
                                        transforms.ToTensor(), _normalize])
_transform_train_cifar10 = transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(),
                                        transforms.RandomErasing(), _normalize])
    
def _get_cifar10_dataloader():
    return CIFAR10(root=f'{_datasets_base_folder}/cifar10', train=True, download=True, transform=_transform_train_cifar10), \
            CIFAR10(root=f'{_datasets_base_folder}/cifar10', train=False, transform=_transform_test_cifar10)

_datasets_get_func['cifar10'] = _get_cifar10_dataloader
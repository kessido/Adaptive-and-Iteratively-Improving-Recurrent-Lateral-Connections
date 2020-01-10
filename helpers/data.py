from functools import partial
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import h5py
import numpy as np


def get_dataloaders(dataset_name, batch_size, num_workers=20):
    trainset, testset = _datasets_get_func[dataset_name]()
    dataloader = partial(torch.utils.data.DataLoader, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return dataloader(trainset), dataloader(testset)
        
    
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
    
def _get_cifar10():
    return CIFAR10(root=f'{_datasets_base_folder}/cifar10', train=True, download=True, transform=_transform_train_cifar10), \
            CIFAR10(root=f'{_datasets_base_folder}/cifar10', train=False, transform=_transform_test_cifar10)

_datasets_get_func['cifar10'] = _get_cifar10

class _H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, x_database_name, y_database_name):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.x_database = x_database_name
            self.y_database = y_database_name
            self.length = len(f[y_database_name])
    
    def __getitem__(self, i):
        with h5py.File(self.h5_path, 'r') as f:
            return f[self.x_database][i], f[self.y_database][i]
        
    def __len__(self):
        return self.length

_cifar10_resnet110_output_dataset_path = f'{_datasets_base_folder}/cifar10/pretrained_resnet110_wo_last_layer_outputs.hdf5'
def _get_cifar10_resnet110_output_dataloader():
    
    return _H5Dataset(_cifar10_resnet110_output_dataset_path, 'train_x', 'train_y'), \
            _H5Dataset(_cifar10_resnet110_output_dataset_path, 'test_x', 'test_y')


_datasets_get_func['cifar10_resnet110_output'] = _get_cifar10_resnet110_output_dataloader

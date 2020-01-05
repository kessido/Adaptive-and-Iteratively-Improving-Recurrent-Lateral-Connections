import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_dataloaders(dataset_name, batch_size, num_workers=20):
    return _datasets_get_func[dataset_name](batch_size, num_workers)

_datasets_base_folder = '/media/data1/idokessler'
_datasets_get_func = {}
    
def _get_cifar10_dataloader(batch_size, num_workers):
    return  torch.utils.data.DataLoader(
                    datasets.CIFAR10(root=f'{datasets_base_folder}/cifar10', train=True, download=True,
                                     batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                                     transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(),
                                        transforms.RandomErasing(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))), \
            torch.utils.data.DataLoader(
                    datasets.CIFAR10(root=f'{datasets_base_folder}/cifar10', train=False, transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])))

_datasets_get_func['cifar10'] = _get_cifar10_dataloader
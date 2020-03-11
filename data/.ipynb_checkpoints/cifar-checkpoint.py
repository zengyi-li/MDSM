import torch
import numpy as np
from torchvision import transforms, datasets


def inf_train_gen(batch_size,flip=True,train=True):
    if flip:
        transf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        
        ])
    else:
        transf = transforms.ToTensor()

    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            '../../data/CIFAR10', train=train, download=True,
            transform=transf
        ), batch_size, drop_last=True,shuffle=True, num_workers=0
    )
    while True:
        for img, labels in loader:
            yield img

            
        
    
    
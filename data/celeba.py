import torch
from torchvision import transforms, datasets


def inf_train_gen(batch_size):
    transf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
        
    ])
    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            'C:/Users/Zengyi/data/32_32_contrast_sharp',
            transform=transf
        ), batch_size, drop_last=True, shuffle=True,pin_memory=True
    )
    while True:
        for img, labels in loader:
            yield img

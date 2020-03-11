import torch
from torchvision import transforms, datasets


def inf_train_gen(batch_size,data_path='default path'):
    transf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
        
    ])
    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            data_path,
            transform=transf
        ), batch_size, drop_last=True, shuffle=True,pin_memory=True
    )
    while True:
        for img, labels in loader:
            yield img

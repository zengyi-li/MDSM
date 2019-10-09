import torch
from torchvision import transforms, datasets


def inf_train_gen_svhn(batch_size):
    transf = transforms.ToTensor()
    
    loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            '../data/SVHN', split='train', download=True,
            transform=transf
        ), batch_size, drop_last=True, shuffle = True
    )
    while True:
        for img, labels in loader:
            yield img
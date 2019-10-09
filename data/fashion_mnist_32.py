import torch
from torchvision import transforms, datasets

def inf_train_gen(batch_size,train=True):
    transf = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
        
    ])
    loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            '../data/fashion_mnist', train=False, download=True,
            transform=transf
        ), batch_size, drop_last=True, shuffle = True
    )
    while True:
        for img, labels in loader:
            yield img

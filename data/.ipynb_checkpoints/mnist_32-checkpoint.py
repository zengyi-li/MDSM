import torch
from torchvision import transforms, datasets


def inf_train_gen(batch_size):
    transf = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
        
    ])
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data/mnist', train=True, download=True,
            transform=transf
        ), batch_size, drop_last=True, shuffle = True
    )
    while True:
        for img, labels in loader:
            yield img
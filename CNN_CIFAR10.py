import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
from MLP_MNIST import Trainer
import time


def accuracy(pred, true):
    _, idxs = torch.max(pred, dim=1)
    return torch.sum(idxs == true).item() / len(idxs)


if __name__ == '__main__':
    train_trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_trans = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset_raw = CIFAR10(root='data/', train=True, download=True, transform=train_trans)
    valid_dataset_raw = CIFAR10(root='data/', train=True, download=True, transform=test_trans)
    test_dataset      = CIFAR10(root='data/', train=False, download=True, transform=test_trans)

    indices = torch.randperm(50000).tolist()
    train_data = Subset(train_dataset_raw, indices[:42000])
    valid_data = Subset(valid_dataset_raw, indices[42000:])

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_data, batch_size=100, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)

    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    simple_model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(256, 10)
    ).to(gpu)

    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    loss_fn = F.cross_entropy

    term = Trainer(simple_model, loss_fn, optimizer, accuracy, device=gpu)
    
    print("Bat dau train...")
    best_model = term.fit(20, train_loader, valid_loader)
    test_loss, test_acc = term.evaluate(test_loader)

    print("Hoan thanh qua trinh huan luyen")
    print("Loss:", test_loss)
    print(f"Accuracy: {test_acc * 100:.2f}%")
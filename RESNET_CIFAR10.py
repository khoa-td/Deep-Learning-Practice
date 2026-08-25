import torch
import torchvision
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as tt
import fastai
from fastai.vision.all import *
import time


def conv_2d(ni, nf, stride= 1, ks= 3):
    return nn.Conv2d(ni, nf, ks, stride, bias= False, padding= ks // 2)


def bn_relu_conv(ni, nf, stride= 1, ks= 3):
    return nn.Sequential(
        nn.BatchNorm2d(ni),
        nn.ReLU(),
        conv_2d(ni, nf, stride, ks)

    )


class ResidualBlock(nn.Module):
    def __init__(self, ni, nf, stride= 1):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d(ni, nf, stride)
        self.conv2 = bn_relu_conv(nf, nf, 1)

        self.shortcut = lambda x: x
        if ni != nf or stride != 1:
            self.shortcut = conv_2d(ni, nf, stride, 1)


    def forward(self, x):
        x = F.relu(self.bn(x), inplace= True)

        r = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x) * 0.2

        return x + r

def make_group(N, ni, nf, stride):
    layers = [ResidualBlock(ni, nf, stride)]

    for _ in range(N - 1):
        layers.append(ResidualBlock(nf, nf, 1))

    return layers

class WideResNet(nn.Module):
    def __init__(self, groups, blocks, k, n_start= 16, n_classes = 10):
        super().__init__()

        layers = [conv_2d(3, n_start)]
        n_chanels = [n_start]

        for i in range(groups):
            chanel_out = (n_start * k) * (2 ** i)
            n_chanels.append(chanel_out)

            stride= 2 if i > 0 else 1
            layers += make_group(blocks, n_chanels[i], n_chanels[i + 1], stride)

        self.body = nn.Sequential(*layers)

        self.tail = nn.Sequential(
            nn.BatchNorm2d(n_chanels[-1]),
            nn.ReLU(inplace= True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(n_chanels[-1], n_classes)
        )

        

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)

        return x


if __name__ == '__main__':

    stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    train_trans = tt.Compose([
        tt.RandomCrop(32, 4, padding_mode= 'reflect'),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    test_valid_trans = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    train_dataset = CIFAR10(root = 'data/', download= True, train= True, transform=train_trans)
    valid_dataset = CIFAR10(root = 'data/', download= True, train= True, transform=test_valid_trans)
    test_dataset = CIFAR10(root = 'data/', download= True, train= False, transform=test_valid_trans)

    indicies = torch.randperm(len(train_dataset)).tolist()
    train_data = Subset(train_dataset, indices=indicies[:42000])
    valid_data = Subset(valid_dataset, indices=indicies[42000:])

    train_loader = DataLoader(train_data, 256, shuffle= True, num_workers= 8, pin_memory= True, persistent_workers= True)
    valid_loader = DataLoader(valid_data, 256, shuffle= False, pin_memory= True)
    test_loader = DataLoader(test_dataset, 256, shuffle= False, pin_memory= True)

    gpu = torch.device('cuda')
    model = WideResNet(3, 4, 6, 16, 10).to(gpu)

    dls = DataLoaders(train_loader, valid_loader, device= gpu)


    learner = Learner(
        dls, 
        model, 
        loss_func=LabelSmoothingCrossEntropyFlat(), 
        metrics=accuracy, 
        cbs=[GradientClip(0.1), MixUp(0.4), SaveModelCallback(monitor='valid_loss', fname='best_wideresnet1')]
    ).to_fp16()

    # lr_res = learner.lr_find(suggest_funcs = (valley, slide))
    learner.fit_one_cycle(10, lr_max=5e-3, wd=1e-4)

    test_loss, test_acc = learner.validate(dl=test_loader)

    print("Test Loss tốt nhất:", test_loss)
    print(f"Test Accuracy tốt nhất: {test_acc * 100:.2f}%")



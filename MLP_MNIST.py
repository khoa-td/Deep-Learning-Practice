import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




class MnistModel(nn.Module):
    def __init__(self, in_size, hiden1, hiden2, out_size):
        super().__init__()
        self.in_ = nn.Linear(in_size, hiden1)
        self.linear2 = nn.Linear(hiden1, hiden2)
        self.out_ = nn.Linear(hiden2, out_size)


    def forward(self, xb):
        xb = torch.flatten(xb, start_dim= 1)
        out = self.in_(xb)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.out_(out)

        return out



class Trainer:
    def __init__(self, model, loss_fn, opt, metric= None, device= None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.opt = opt
        self.metric = metric

    def loss_batch(self, xb, yb, is_train=True):
        xb, yb = xb.to(self.device), yb.to(self.device)

        y_pred = self.model(xb)
        loss = self.loss_fn(y_pred, yb)

        if is_train and self.opt is not None:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        metric_res = self.metric(y_pred, yb) if self.metric else None

        return loss.item(), len(xb), metric_res


    def evaluate(self, dataloader):
        self.model.eval()
        sum_loss, count, sum_metric = 0, 0, 0

        with torch.no_grad():
            for x, y in dataloader:
                term_loss, term_count, term_metric = self.loss_batch(x, y, is_train = False)
                sum_loss += term_loss * term_count
                count += term_count
                if term_metric is not None:
                    sum_metric += term_metric * term_count


        return sum_loss / count, sum_metric / count if self.metric else None


    def fit(self, epochs, train_ld, valid_ld):
        best_model_state = None
        best_loss = float('inf')
        start_time = time.time()

        for _ in range(epochs):
            self.model.train()
            for xb, yb in train_ld:
                self.loss_batch(xb, yb, is_train= True)

            val_loss, val_acc = self.evaluate(valid_ld)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict()) # chi luu cac trong so cua model

        print(f"Thoi gian hoan thanh huan luyen:{time.time() - start_time:.2f}s")
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return self.model

if __name__ == '__main__':

    
    dataset = MNIST(root = 'data/', download= True)
    test_data = MNIST(root = 'data/', train= False, transform=transforms.ToTensor())
    dataset = MNIST(root = 'data/', train= True, transform= transforms.ToTensor())


    train_data, valid_data = random_split(dataset, [50000, 10000])

    train_loader = DataLoader(train_data, 128, shuffle= True)
    valid_loader = DataLoader(valid_data, 128, shuffle= False)
    test_loader = DataLoader(test_data, 128, shuffle= False)

    img, label = dataset[0]
    input_size = img.numel()

    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MnistModel(input_size, 64, 32, 10).to(device_)
    
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    def accuracy(pred, true):
        _, idx = torch.max(pred, dim = 1)

        return torch.sum(idx== true).item() / len(idx)

    term = Trainer(model, loss_fn, optimizer, accuracy, device_)
    best_model = term.fit(50, train_loader, valid_loader)

    test_loss, test_acc = term.evaluate(test_loader)

    print(f"Do chinh xac tren tap test | Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%")

    img, label = test_data[4176]

    xb = img.unsqueeze(0).to(device_)
    y_pred = best_model(xb)
    label_pred = torch.argmax(y_pred, dim = 1).item()
    print("Du doan anh thu 4176:", label_pred)
    print("Ket qua chinh xac:", label)

    plt.figure(figsize=(4, 3))
    plt.imshow(img.squeeze(), cmap = 'Blues')
    plt.axis('off')
    plt.title("Buc anh thu 4176")
    plt.savefig('anh_tam.png', bbox_inches='tight')



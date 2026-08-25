import torch
import torchvision
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


dataset = MNIST(root = 'data/', download = True)

test_dataset = MNIST(root = 'data/', train = False, transform= transforms.ToTensor())
dataset = MNIST(root = 'data/', train = True, transform = transforms.ToTensor())


def train_split(n, valid_rate):
    pivot = int(n * valid_rate)

    idxs = np.random.permutation(n)
    return idxs[pivot:], idxs[:pivot]

train_idxs, valid_idxs = train_split(len(dataset), valid_rate = 0.2)

train_sampler = SubsetRandomSampler(train_idxs)
train_loader = DataLoader(dataset, 100, sampler = train_sampler)

valid_sampler = SubsetRandomSampler(valid_idxs)
valid_loader = DataLoader(dataset, 100, sampler = valid_sampler)

input_size = 28 * 28
num_classes = 10

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        
        self.out = nn.Linear(64, 10)
    
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        xb = F.relu(self.fc3(xb))
        
        return self.out(xb)

model = MnistModel()
loss_fn = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

def accuracy_sc(pred, label):
    _, y_pred = torch.max(pred, dim = 1)
    return torch.sum(y_pred == label).item()/ len(y_pred)

def loss_batch(model, loss_fn, xb, yb, opt= None, metric= None):
    y_pred = model(xb)
    loss = loss_fn(y_pred, yb)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    
    metric_res = None
    if metric is not None:
        metric_res = metric(y_pred, yb)
    
    return loss.item(), len(xb), metric_res


def evaluate(model, loss_fn, valid_dl, metric = None):
    with torch.no_grad():
        sum_loss = 0
        count_batch = 0
        sum_metric = 0
        for xb, yb in valid_dl:
            term = loss_batch(model, loss_fn, xb, yb, metric= metric)
            sum_loss += (term[0] * term[1])
            count_batch += term[1]
            sum_metric += (term[2] * term[1])
        avg_loss = sum_loss / count_batch
        avg_metric = None
        if metric is not None:
            avg_metric = sum_metric / count_batch
        
    return avg_loss, count_batch, avg_metric


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric= None):
    best_model = None
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()

        for xb, yb in train_dl:
            term = loss_batch(model, loss_fn, xb, yb, opt, metric)
        
        model.eval()
        loss_value, _, acc_score = evaluate(model, loss_fn, valid_dl, metric)

        if loss_value < best_loss:
            best_loss = loss_value
            best_model = copy.deepcopy(model)
    return best_model

best_model = fit(20, model, loss_fn, optimizer, train_loader, valid_loader, accuracy_sc)

test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

test_loss, _, test_acc = evaluate(best_model, loss_fn, test_loader, metric=accuracy_sc)

print(f"Do chinh xac tren tap test | Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%")

img, true_label = test_dataset[6789]
xb = img.unsqueeze(0)

pred = torch.argmax(best_model(xb), dim=1).item()

plt.imshow(img.squeeze(), cmap = 'Blues')
plt.title("Buc anh thu 4176")
plt.axis('off')
plt.savefig("ket_qua_du_doan.png")

print("Du doan buc anh thu 4176:", pred)
print("Ket qua chinh xac la:", true_label)
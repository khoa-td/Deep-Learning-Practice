import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

df = pd.read_csv('khach_hang_mua_sgd_sach.csv')
df['kenh_tiep_can'] = df['kenh_tiep_can'].fillna(df['kenh_tiep_can'].mode()[0])
df['khu_vuc'] = df['khu_vuc'].fillna(df['khu_vuc'].mode()[0])

for col in ['tuoi', 'thu_nhap_trieu', 'so_lan_mua_truoc', 'so_ngay_tu_lan_mua_cuoi', 'diem_tin_dung']:
    df[col] = pd.to_numeric(df[col], errors = 'coerce')
    df[col] = df[col].fillna(df[col].median())
df = pd.get_dummies(df, columns=['kenh_tiep_can', 'khu_vuc'], drop_first=True)

X = df.drop(columns = 'mua_hang').values
y = df['mua_hang'].values

X_train, X_term, y_train, y_term = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_valid, X_test, y_valid, y_test = train_test_split(X_term, y_term, test_size = 0.5, random_state = 42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


#Chuyen sang du lieu cua PyTorch:
X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_valid = torch.FloatTensor(y_valid).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)


#chay neural networks cua deeplearning:
class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.func1 = nn.Linear(input_dim, 16)
        self.func2 = nn.Linear(16, 8)
        self.func3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.func1(x))
        x = self.drop(x)
        x = self.relu(self.func2(x))
        x = self.sig(self.func3(x))

        return x

#train model vua tao:
model = Model(input_dim = X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

MAX_EPOCHS = 200
PATIENCE = 10
patience_count = 0
best_valid_loss = float('inf')
best_model_state = None

for epoch in range(MAX_EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)

    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_outputs = model(X_valid)
        valid_loss = criterion(valid_outputs, y_valid)
    
    if valid_loss.item() < best_valid_loss:
        best_valid_loss = valid_loss.item()
        best_model_state = copy.deepcopy(model.state_dict())
        patience_count = 0

    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            break

model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    y_pred = (test_outputs >= 0.5).float()
accuracy = float(accuracy_score(y_test.numpy(), y_pred.numpy()) * 100)
print(f"Accuracy: {accuracy:.2f}%")

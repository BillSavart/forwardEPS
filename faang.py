import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# preprocessing the data in train/FAANG.csv
df = pd.read_csv('train/FAANG.csv')
df = df[['Ticker', 'Date', 'Forward PE', 'Volume', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by=['Ticker', 'Date'], inplace=True)
df['Prev Close'] = df.groupby('Ticker')['Close'].shift(1)
df = df.dropna()

# data before 2023 is used for training and data after 2023 is used for validation
train_data = df[df['Date'].dt.year <= 2023]
val_data = df[df['Date'].dt.year > 2023]

# prepare the data for training and validation
X_train = train_data[['Forward PE', 'Volume', 'Prev Close']].values
y_train = train_data['Close'].values.reshape(-1, 1)
X_val = val_data[['Forward PE', 'Volume', 'Prev Close']].values
y_val = val_data['Close'].values.reshape(-1, 1)

# Scale the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_val_scaled = scaler_x.transform(X_val)
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)

# Create a custom dataset
class FAANGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Create DataLoader
train_dataset = FAANGDataset(X_train_scaled, y_train_scaled)
val_dataset = FAANGDataset(X_val_scaled, y_val_scaled)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the model
class FAANGModel(nn.Module):
    def __init__(self):
        super().__init__()
        nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.x
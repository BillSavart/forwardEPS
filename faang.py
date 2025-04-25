import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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


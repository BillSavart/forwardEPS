import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === Step 1: 抓取台積電歷史股價 ===
ticker = yf.Ticker("2330.TW")
hist = ticker.history(period="10y")
hist.reset_index(inplace=True)
hist = hist[["Date", "Close"]]
hist.columns = ["Date", "StockPrice"]

# === Step 2: 模擬 Forward PE 與 EPS 數據（實務應每日抓） ===
np.random.seed(42)
hist["ForwardPE"] = np.random.uniform(10, 20, len(hist))
hist["ForwardEPS"] = np.random.uniform(15, 25, len(hist))

# === Step 3: 計算未來6個月後的報酬率 ===
# 用交易日來計算，126個交易日大約是6個月
hist["FuturePrice"] = hist["StockPrice"].shift(-126)
hist["6MonthReturn"] = (hist["FuturePrice"] - hist["StockPrice"]) / hist["StockPrice"]
hist.dropna(inplace=True)  # 移除結尾無法計算報酬率的資料

# === Step 4: 建立訓練資料集 ===
X = hist[["ForwardPE", "ForwardEPS", "StockPrice"]].values
y = hist["6MonthReturn"].values.reshape(-1, 1)

scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# === Step 5: PyTorch 模型 ===
class ReturnPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = ReturnPredictor()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 訓練模型
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# === Step 6: 預測與評估 ===
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = scaler_y.inverse_transform(predictions.numpy())
    actuals = scaler_y.inverse_transform(y_test.numpy())

# 畫圖
plt.scatter(actuals, predictions)
plt.xlabel("Actual 6-Month Return")
plt.ylabel("Predicted 6-Month Return")
plt.title("Prediction vs Actual")
plt.plot([-0.5, 0.5], [-0.5, 0.5], 'r--')
plt.grid(True)
plt.show()

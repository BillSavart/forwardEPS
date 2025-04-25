import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀你的資料
df = pd.read_csv('train/FAANG.csv')
df = df[['Ticker', 'Date', 'EPS', 'Volume', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by=['Ticker', 'Date'], inplace=True)
df['Prev Close'] = df.groupby('Ticker')['Close'].shift(1)
df = df.dropna()
df['EPS'] = df['Close'] / df['EPS']

# 選一家股票，例如 AAPL
df_aapl = df[df['Ticker'] == 'AAPL']

# 1. 時間序列圖
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df_aapl['Date'], df_aapl['Close'], label='Close Price', color='blue')
ax1.set_ylabel('Close Price', color='blue')
ax2 = ax1.twinx()
ax2.plot(df_aapl['Date'], df_aapl['EPS'], label='EPS', color='green', linestyle='--')
ax2.set_ylabel('EPS', color='green')
plt.title('示意圖1: Close vs EPS')
plt.show()

# 2. 散佈圖：EPS vs Close
plt.figure(figsize=(8, 6))
sns.scatterplot(x='EPS', y='Close', data=df_aapl)
plt.title('示意圖2: EPS vs Close')
plt.show()

# 3. 散佈圖：Volume vs Close
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Volume', y='Close', data=df_aapl)
plt.title('示意圖3: Volume vs Close')
plt.show()

# 4. 相關係數熱力圖
corr = df_aapl[['EPS', 'Volume', 'Close']].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('示意圖4: Correlation Matrix')
plt.show()

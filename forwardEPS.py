import yfinance as yf
import pandas as pd

# 台灣50成分股代碼
stock_codes = [
    "2330.TW","AAPL","NVDA","TSLA","AMZN","GOOGL","MSFT","TSM","META","NFLX",
    "INTC","AMD","CSCO","ORCL","IBM","QCOM","AVGO","TXN","ADBE","CRM"
]

# 抓取每檔股票的市值與 PE 資訊
stocks_data = []
for code in stock_codes:
    stock = yf.Ticker(code)
    info = stock.info

    stocks_data.append({
        "股票代碼": code.replace(".TW", ""),
        "公司名稱": info.get("shortName", "N/A"),
        "Forward EPS": info.get("forwardEps", "N/A"),
        "Trailing EPS": info.get("trailingEps", "N/A")
    })

# 建立 DataFrame 並依市值排序
df = pd.DataFrame(stocks_data)

# 印出資料
print(df)
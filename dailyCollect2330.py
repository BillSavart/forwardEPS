import yfinance as yf
import pandas as pd
import datetime
import os

def collect_and_save():
    today = datetime.date.today()
    ticker = yf.Ticker("2330.TW")
    info = ticker.info
    hist = ticker.history(period="1d")

    if hist.empty:
        print("今天無交易資料")
        return

    price = hist["Close"].iloc[-1]
    forward_pe = info.get("forwardPE", None)
    forward_eps = info.get("forwardEps", None)

    data = pd.DataFrame([{
        "Date": today,
        "StockPrice": price,
        "ForwardPE": forward_pe,
        "ForwardEPS": forward_eps
    }])

    file = "train/2330_daily_data.csv"
    header = not os.path.exists(file)

    data.to_csv(file, mode='a', header=header, index=False)
    print(f"寫入成功：{data}")

if __name__ == "__main__":
    collect_and_save()

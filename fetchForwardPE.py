import yfinance as yf

def get_PE(stock_code):
    """
    Fetches the Forward PE for a given stock code.
    
    Args:
        stock_code (str): The stock code to fetch PE for.
        
    Returns:
        Forward PE (or "N/A" if not available).
    """
    stock = yf.Ticker(stock_code)
    return stock.info.get("forwardPE", "N/A")

def __main__():
    stock = yf.Ticker("2330.TW")
    print(stock.info.get("forwardPE", "N/A"))
    
if __name__ == "__main__":
    __main__()
    

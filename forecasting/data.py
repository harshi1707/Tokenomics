import pandas as pd
import numpy as np


def download_ohlc(ticker: str = 'BTC-USD', period: str = '5y', interval: str = '1d') -> pd.DataFrame:
    try:
        import yfinance as yf  # lazy import so API can run without optional dep
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        df = df[['Close']].dropna().reset_index()
        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        return df
    except Exception:
        # Fallback: generate a synthetic random walk as placeholder data
        num_points = 365 * 3
        rng = np.random.default_rng(seed=42)
        steps = rng.normal(loc=0.0, scale=1.0, size=num_points)
        prices = 100 + np.cumsum(steps)
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=num_points, freq='D')
        return pd.DataFrame({'ds': dates, 'y': prices})


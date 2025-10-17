import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from forecasting.data import download_ohlc
from forecasting.models import LSTMForecaster, GRUForecaster, make_sequences
from coingecko_service import coingecko_service


def get_coingecko_ohlc(coin_id: str, days: int = 365) -> pd.DataFrame:
    """Download OHLC data for a coin from CoinGecko and format for forecasting"""
    try:
        # Get historical prices from CoinGecko
        data = coingecko_service.get_historical_prices(coin_id, days=days, vs_currency='usd')

        if 'prices' not in data:
            raise ValueError("No price data available")

        # Extract prices: prices is list of [timestamp, price]
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['ds'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['y'] = df['price']
        df = df[['ds', 'y']].dropna().reset_index(drop=True)

        return df
    except Exception as e:
        # Fallback to yfinance if CoinGecko fails
        print(f"CoinGecko data fetch failed: {e}, falling back to yfinance")
        # Convert coin_id to ticker format (e.g., 'bitcoin' -> 'BTC-USD')
        ticker_map = {
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD',
            'tether': 'USDT-USD',
            'binancecoin': 'BNB-USD',
            'cardano': 'ADA-USD',
            'solana': 'SOL-USD',
            'polkadot': 'DOT-USD',
            'dogecoin': 'DOGE-USD',
            'avalanche-2': 'AVAX-USD',
            'polygon': 'MATIC-USD'
        }
        ticker = ticker_map.get(coin_id, f"{coin_id.upper()}-USD")
        return download_ohlc(ticker=ticker, period=f"{days//30}mo", interval='1d')


def generate_forecast(coin_id: str, horizon: int = 7, model_type: str = 'lstm', lookback: int = 30) -> dict:
    """Generate price forecast with confidence intervals for a cryptocurrency"""
    try:
        # Get historical data
        df = get_coingecko_ohlc(coin_id, days=max(365, lookback + horizon + 10))

        if len(df) < lookback + horizon:
            return {
                'error': 'Not enough historical data for forecasting',
                'coin_id': coin_id,
                'available_data_points': len(df)
            }

        # Prepare sequences
        X, y = make_sequences(df['y'], lookback=lookback, horizon=horizon)

        if len(X) < 5:  # Need minimum data for training
            return {
                'error': 'Insufficient data for model training',
                'coin_id': coin_id,
                'sequences_created': len(X)
            }

        device = 'cpu'
        ModelClass = LSTMForecaster if model_type.lower() == 'lstm' else GRUForecaster
        model = ModelClass(horizon=horizon).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        # Split data for training and validation
        X_train, y_train = X[:-2].to(device), y[:-2].to(device)
        X_val, y_val = X[-2:].to(device), y[-2:].to(device)
        X_last = X[-1:].to(device)

        # Train model
        model.train()
        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(50):  # More epochs for better training
            optimizer.zero_grad()
            preds = model(X_train.float())
            loss = loss_fn(preds, y_train.float())
            loss.backward()
            optimizer.step()

            # Early stopping based on validation loss
            if len(X_val) > 0:
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val.float())
                    val_loss = loss_fn(val_preds, y_val.float())
                model.train()

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        # Generate forecast
        model.eval()
        with torch.no_grad():
            forecast_vals = model(X_last.float()).cpu().numpy().reshape(-1).tolist()

        # Calculate confidence intervals using prediction variance
        # For simplicity, use a percentage-based confidence interval
        # In a real implementation, you'd use Monte Carlo dropout or ensemble methods
        current_price = df['y'].iloc[-1]
        forecast_std = np.std(df['y'].tail(30))  # Use recent volatility as proxy

        confidence_intervals = []
        for i, pred in enumerate(forecast_vals):
            # Increasing uncertainty over time
            uncertainty_factor = 1 + (i * 0.1)  # 10% increase per day
            std = forecast_std * uncertainty_factor

            lower = max(0, pred - 1.96 * std)  # 95% confidence
            upper = pred + 1.96 * std

            confidence_intervals.append({
                'day': i + 1,
                'predicted_price': pred,
                'lower_bound': lower,
                'upper_bound': upper,
                'confidence_level': 0.95
            })

        # Generate future dates
        last_date = df['ds'].iloc[-1]
        future_dates = [(last_date + timedelta(days=i+1)).isoformat() for i in range(horizon)]

        return {
            'coin_id': coin_id,
            'model_type': model_type.upper(),
            'horizon': horizon,
            'lookback': lookback,
            'current_price': current_price,
            'forecast': {
                'dates': future_dates,
                'predictions': confidence_intervals
            },
            'metadata': {
                'training_sequences': len(X_train),
                'data_points': len(df),
                'last_updated': datetime.now().isoformat()
            }
        }

    except Exception as e:
        return {
            'error': f'Forecasting failed: {str(e)}',
            'coin_id': coin_id
        }
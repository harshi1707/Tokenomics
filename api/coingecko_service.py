import requests
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class CoinGeckoService:
    BASE_URL = "https://api.coingecko.com/api/v3"
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self):
        self.session = requests.Session()
        # CoinGecko free tier: 10-50 calls/minute, 1000 calls/day
        self.last_request_time = None
        self.min_request_interval = 0.1  # 100ms between requests to be safe

    def _make_request(self, endpoint, params=None):
        """Make a request to CoinGecko API with rate limiting"""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

        url = f"{self.BASE_URL}{endpoint}"
        try:
            logger.debug(f"Making request to: {url} with params: {params}")
            response = self.session.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
            self.last_request_time = time.time()

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded, retrying after delay")
                time.sleep(1)  # Wait 1 second and retry
                return self._make_request(endpoint, params)
            else:
                logger.error(f"CoinGecko API error: {response.status_code} - {response.text}")
                raise Exception(f"API Error: {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Network error: {str(e)}")

    def get_market_data(self, vs_currency='usd', order='market_cap_desc', per_page=250, page=1, sparkline=False):
        """Fetch market data for cryptocurrencies"""
        params = {
            'vs_currency': vs_currency,
            'order': order,
            'per_page': per_page,
            'page': page,
            'sparkline': str(sparkline).lower()
        }
        return self._make_request('/coins/markets', params)

    def get_coin_details(self, coin_id):
        """Fetch detailed information about a specific coin"""
        return self._make_request(f'/coins/{coin_id}')

    def get_historical_prices(self, coin_id, days=1, vs_currency='usd'):
        """Fetch historical price data for a coin"""
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        return self._make_request(f'/coins/{coin_id}/market_chart', params)

# Global instance
coingecko_service = CoinGeckoService()
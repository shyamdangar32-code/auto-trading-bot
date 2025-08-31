# strategies/intraday_rsi.py
import pandas as pd

class IntradayRSIStrategy:
    """
    Simple Intraday RSI Strategy
    """

    def __init__(self, rsi_period=14, overbought=70, oversold=30):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    def calculate_rsi(self, prices: pd.Series):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signal(self, prices: pd.Series):
        rsi = self.calculate_rsi(prices)
        latest_rsi = rsi.iloc[-1]

        if latest_rsi > self.overbought:
            return "SELL"
        elif latest_rsi < self.oversold:
            return "BUY"
        else:
            return "HOLD"

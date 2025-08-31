# broker/zerodha.py
from kiteconnect import KiteConnect

class ZerodhaClient:
    """
    Simple Zerodha client wrapper around kiteconnect.KiteConnect
    """

    def __init__(self, api_key: str, access_token: str = None):
        self.kite = KiteConnect(api_key=api_key)
        if access_token:
            self.kite.set_access_token(access_token)

    def get_profile(self):
        """Fetch Zerodha account profile"""
        return self.kite.profile()

    def get_quote(self, instruments):
        """Fetch live market quotes"""
        return self.kite.quote(instruments)

    def place_order(self, tradingsymbol, qty, transaction_type, order_type="MARKET", product="MIS"):
        """Place simple market order"""
        return self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NSE,
            tradingsymbol=tradingsymbol,
            transaction_type=transaction_type,
            quantity=qty,
            order_type=order_type,
            product=product
        )

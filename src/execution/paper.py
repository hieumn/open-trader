# minimal paper execution adapter (no real broker)
import logging

logger = logging.getLogger(__name__)

class PaperBroker:
    def __init__(self):
        self.positions = {}

    def send_order(self, symbol: str, qty: int, side: str, price: float):
        logger.info("Paper order: %s %s %d @ %s", side, symbol, qty, price)
        # immediate fill
        if side.lower() == 'buy':
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
        else:
            self.positions[symbol] = self.positions.get(symbol, 0) - qty
        return {"status":"filled","filled_qty":qty,"avg_fill":price}
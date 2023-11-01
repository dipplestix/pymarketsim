from ..fundamental.fundamental import Fundamental
from ..event.event_queue import EventQueue


class Market:
    def __init__(self, fundamental: Fundamental):
        self.order_book = FourHeap()
        self.matched_orders = []
        self.fundamental = Fundamental
        self.event_queue

    def withdraw_all(self, agent_id):
        self.order_book.withdraw_all(agent_id)

    def clear(self):
        new_orders = self.order_book.market_clear()
        self.matched_orders += new_orders


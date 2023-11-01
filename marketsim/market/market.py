from typing import List
from .fundamental import Fundamental

class Market:
    def __init__(self):
        self.order_book = FourHeap()
        self.matched_orders = []

    def withdraw_all(self, agent_id):
        self.order_book.withdraw_all(agent_id)

    def clear(self):
        new_orders = self.order_book.market_clear()
        self.matched_orders += new_orders


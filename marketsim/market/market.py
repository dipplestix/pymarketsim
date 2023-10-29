from typing import List


class Market:
    def __init__(self):
        self.order_book = FourHeap()
        self.matched_orders = []

    def withdraw_all(self, agent_id):
        self.order_book.withdraw_all(agent_id)

    def clear(self) -> List[Order]:
        new_orders = self.order_book.market_clear()
        self.matched_orders.extend(new_orders)

        return new_orders

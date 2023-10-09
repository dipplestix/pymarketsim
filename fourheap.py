from order_queue import OrderQueue
from order import Order
import constants


class FourHeap:
    def __init__(self):
        self.buy_matched = OrderQueue(is_max_heap=False, is_matched=True)
        self.buy_unmatched = OrderQueue(is_max_heap=True, is_matched=False)
        self.sell_matched = OrderQueue(is_max_heap=True, is_matched=True)
        self.sell_unmatched = OrderQueue(is_max_heap=False, is_matched=False)

        self.heaps = [self.buy_matched, self.buy_unmatched, self.sell_matched, self.sell_unmatched]

        self.agent_id_map = {}

    def insert(self, order: Order):
        if order.order_type == constants.SELL:
            if order.price <= self.buy_unmatched.peek() and self.sell_matched.peek() <= self.buy_unmatched.peek():
                self.sell_matched.add_order(order)
                b = self.buy_unmatched.push_to()
                self.buy_matched.add_order(b)
            elif order.price < self.sell_matched.peek():
                s = self.sell_matched.push_to()
                self.sell_matched.add_order(order)
                self.sell_unmatched.add_order(s)
            else:
                self.sell_unmatched.add_order(order)
        if order.order_type == constants.BUY:
            if order.price >= self.sell_unmatched.peek() and self.buy_matched.peek() >= self.sell_unmatched.peek():
                self.buy_matched.add_order(order)
                s = self.sell_unmatched.push_to()
                self.sell_matched.add_order(s)
            elif order.price > self.buy_matched.peek():
                b = self.buy_matched.push_to()
                self.buy_matched.add_order(order)
                self.buy_unmatched.add_order(b)
            else:
                self.buy_unmatched.add_order(order)

    def remove(self, order_id: int):
        pass

    def withdraw_all(self, agent_id: int):
        for order_id in self.agent_id_map[agent_id]:
            self.remove(order_id)

    def market_clear(self):
        pass

    def get_bid_quote(self) -> float:
        return max(self.buy_unmatched.peek(), self.sell_matched.peek())

    def get_ask_quote(self) -> float:
        return max(self.sell_unmatched.peek(), self.buy_matched.peek())

    def observe(self) -> str:
        s = '--------------\n'
        names = ['buy_matched', 'buy_unmatched', 'sell_matched', 'sell_unmatched']
        for i, heap in enumerate(self.heaps):
            s += names[i]
            s += '\n'
            s += f'Top order_id: {heap.peek_order().order_id}\n'
            s += f'Top price: {abs(heap.peek())}\n'
            s += f'Number of orders: {heap.count()}\n'

        return s

from collections import defaultdict

import constants
from order import Order, MatchedOrder
from order_queue import OrderQueue


class FourHeap:
    def __init__(self):
        self.buy_matched = OrderQueue(is_max_heap=False, is_matched=True)
        self.buy_unmatched = OrderQueue(is_max_heap=True, is_matched=False)
        self.sell_matched = OrderQueue(is_max_heap=True, is_matched=True)
        self.sell_unmatched = OrderQueue(is_max_heap=False, is_matched=False)

        self.heaps = [self.buy_matched, self.buy_unmatched, self.sell_matched, self.sell_unmatched]
        self.agent_id_map = defaultdict(list)

    def insert(self, order: Order):
        self.agent_id_map[order.agent_id].append(order.order_id)
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
        if self.buy_unmatched.contains(order_id):
            self.buy_unmatched.remove(order_id)
        elif self.sell_unmatched.contains(order_id):
            self.sell_unmatched.remove(order_id)
        elif self.buy_matched.contains(order_id):
            self.buy_matched.remove(order_id)
            s = self.sell_matched.push_to()
            self.insert(s)
        elif self.sell_matched.contains(order_id):
            self.sell_matched.remove(order_id)
            b = self.buy_matched.push_to()
            self.insert(b)

    def withdraw_all(self, agent_id: int):
        for order_id in self.agent_id_map[agent_id]:
            self.remove(order_id)

    def market_clear(self, plus_one=False):
        matched_count = self.buy_matched.count()
        b_i = 0
        s_i = 0
        p = self.get_ask_quote() if plus_one else self.get_bid_quote()
        matched_orders = []
        while matched_count > 0:
            b_order = self.buy_matched.heap[b_i]
            s_order = self.sell_matched.heap[s_i]
            while b_order.order_id in self.buy_matched.deleted_ids:
                b_i += 1
                b_order = self.buy_matched.heap[b_i]
            while s_order.order_id in self.sell_matched.deleted_ids:
                s_i += 1
                s_order = self.sell_matched.heap[s_i]

            matched_order = MatchedOrder(price=p, quantity=1, buy_order=b_order, sell_order=s_order)
            matched_orders.append(matched_order)
            matched_count -= 1

        self.buy_matched.clear()
        self.sell_matched.clear()
        return matched_orders

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
            s += f'Number of orders: {heap.count()}\n\n\n'

        return s

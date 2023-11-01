from order_queue import OrderQueue
from order import Order
import constants
from collections import defaultdict


class FourHeap:
    def __init__(self):
        self.buy_matched = OrderQueue(is_max_heap=False, is_matched=True)
        self.buy_unmatched = OrderQueue(is_max_heap=True, is_matched=False)
        self.sell_matched = OrderQueue(is_max_heap=True, is_matched=True)
        self.sell_unmatched = OrderQueue(is_max_heap=False, is_matched=False)

        self.heaps = [self.buy_matched, self.buy_unmatched, self.sell_matched, self.sell_unmatched]
        self.agent_id_map = defaultdict(list)

    def handle_new_order(self, order):
        q_order = order.quantity
        order_matched = self.sell_matched if order.order_type == constants.SELL else self.buy_matched
        counter_matched = self.sell_matched if order.order_type == constants.BUY else self.buy_matched
        counter_unmatched = self.sell_unmatched if order.order_type == constants.BUY else self.buy_unmatched

        to_match = counter_unmatched.push_to()
        if to_match is not None:
            to_match_quantity = to_match.quantity
            if to_match_quantity == q_order:
                order_matched.add_order(order)
                counter_matched.add_order(to_match)
            elif to_match_quantity > q_order:
                unmatched_b = to_match.copy_and_decrease(q_order)
                order_matched.add_order(order)
                counter_matched.add_order(to_match)
                counter_unmatched.add_order(unmatched_b)
            elif q_order > to_match_quantity:
                # There's a better way to do this, but I think it's not worth it
                counter_matched.add_order(to_match)
                new_order = order.copy_and_decrease(to_match_quantity)
                order_matched.add_order(order)
                self.insert(new_order)

    def handle_replace(self, order):
        matched = self.sell_matched if order.order_type == constants.SELL else self.buy_matched
        unmatched = self.sell_unmatched if order.order_type == constants.SELL else self.buy_unmatched
        q_order = order.quantity
        replaced = matched.push_to()
        if replaced is not None:
            replaced_quantity = replaced.quantity
            if replaced_quantity == q_order:
                matched.add_order(order)
                unmatched.add_order(replaced)
            elif replaced_quantity > q_order:
                matched.add_order(order)
                matched_s = replaced.copy_and_decrease(q_order)
                matched.add_order(matched_s)
                unmatched.add_order(replaced)
            elif replaced_quantity < q_order:
                unmatched.add_order(replaced)
                new_order = order.copy_and_decrease(replaced_quantity)
                matched.add_order(order)
                self.insert(new_order)

    def insert(self, order: Order):
        self.agent_id_map[order.agent_id].append(order.order_id)
        if order.order_type == constants.SELL:
            if order.price <= self.buy_unmatched.peek() and self.sell_matched.peek() <= self.buy_unmatched.peek():
                self.handle_new_order(order)
            elif order.price < self.sell_matched.peek():
                self.handle_replace(order)
            else:
                self.sell_unmatched.add_order(order)
        elif order.order_type == constants.BUY:
            if order.price >= self.sell_unmatched.peek() and self.buy_matched.peek() >= self.sell_unmatched.peek():
                self.handle_new_order(order)
            elif order.price > self.buy_matched.peek():
                self.handle_replace(order)
            else:
                self.buy_unmatched.add_order(order)

    def remove(self, order_id: int):
        if self.buy_unmatched.contains(order_id):
            self.buy_unmatched.remove(order_id)
        elif self.sell_unmatched.contains(order_id):
            self.sell_unmatched.remove(order_id)
        elif self.buy_matched.contains(order_id):
            order_q = self.buy_matched.order_dict[order_id].quantity
            self.buy_matched.remove(order_id)
            s = self.sell_matched.push_to()
            s_quantity = s.quantity
            if s_quantity >= order_q:
                self.insert(s)
            elif s_quantity < order_q:
                while order_q > 0:
                    order_q -= s_quantity
                    self.insert(s)
                    s = self.sell_matched.push_to()
                    s_quantity = s.quantity
        elif self.sell_matched.contains(order_id):
            order_q = self.sell_matched.order_dict[order_id].quantity
            self.sell_matched.remove(order_id)
            b = self.buy_matched.push_to()
            b_quantity = b.quantity
            if b_quantity >= order_q:
                self.insert(b)
            elif b_quantity < order_q:
                while order_q > 0:
                    order_q -= b_quantity
                    self.insert(b)
                    b = self.buy_matched.push_to()
                    b_quantity = b.quantity

    def withdraw_all(self, agent_id: int):
        for order_id in self.agent_id_map[agent_id]:
            self.remove(order_id)

    def market_clear(self, plus_one=False):
        p = self.get_ask_quote() if plus_one else self.get_bid_quote()

        buy_matched = self.buy_matched.market_clear(p)
        sell_matched = self.sell_matched.market_clear(p)

        matched_orders = buy_matched + sell_matched

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

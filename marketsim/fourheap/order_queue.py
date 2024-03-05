import heapq
import math
from typing import Optional

from order import Order, MatchedOrder


class OrderQueue:
    def __init__(self, is_max_heap=False, is_matched=False):
        self.is_max_heap = is_max_heap
        self.is_matched = is_matched

        self.size = 0
        self.heap = []
        self.order_dict = {}
        self.deleted_ids = set()

    def add_order(self, order: Order):
        price = order.price if not self.is_max_heap else -order.price
        order_id = order.order_id
        if self.contains(order_id):
            self.order_dict[order_id].merge_order(order.quantity)
        else:
            heapq.heappush(self.heap, (price, order.order_id))
            self.order_dict[order.order_id] = order
        self.size += order.quantity

    def peek(self) -> float:
        c = -1 if self.is_max_heap else 1

        if self.is_empty():
            return c*math.inf

        return c*self.heap[0][0]

    def peek_order(self) -> Order:
        if self.is_empty():
            return None
            # return Order(price=0, agent_id=0, order_id=0, order_type=0, quantity=0, time=0)
        order_id = self.heap[0][1]
        return self.order_dict[order_id]

    def peek_order_id(self) -> float:
        return self.heap[0][1]

    def clear(self):
        self.heap = []
        self.order_dict = {}
        self.deleted_ids = set()
        self.size = 0

    def market_clear(self, p):
        if self.is_matched:
            matched_orders = []
            for _, order_id in self.heap:
                if order_id not in self.deleted_ids:
                    order = self.order_dict[order_id]
                    matched_orders.append(MatchedOrder(p, order))
            self.clear()
            return matched_orders
        return None

    def is_empty(self) -> bool:
        return self.size == 0 or len(self.heap) == 0

    def count(self) -> int:
        return self.size

    def remove(self, order_id: int):
        if self.contains(order_id):
            self.deleted_ids.add(order_id)
            self.size -= self.order_dict[order_id].quantity
        try:
            while self.peek_order().order_id in self.deleted_ids:
                heapq.heappop(self.heap)
        except (KeyError, AttributeError):
            pass
        del self.order_dict[order_id]

    def contains(self, order_id: int) -> bool:
        return order_id in self.order_dict

    def push_to(self) -> Optional['Order']:
        while self.heap:
            price, order_id = heapq.heappop(self.heap)
            if order_id not in self.deleted_ids:
                order = self.order_dict[order_id]
                self.size -= order.quantity
                del self.order_dict[order.order_id]

                # Make sure the new top of heap is not a removed order
                try:
                    while self.peek_order_id in self.deleted_ids:
                        heapq.heappop(self.heap)
                except IndexError:
                    pass

                return order
        return None

    def __str__(self):
        s = ''
        for _, order_id in self.heap:
            if order_id not in self.deleted_ids:
                order = self.order_dict[order_id]
                s += str(order) + '\n'

        return s

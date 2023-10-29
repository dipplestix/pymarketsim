import heapq
from typing import Optional
from order import Order
import math


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
        heapq.heappush(self.heap, (price, order))
        self.order_dict[order.order_id] = order
        self.size += order.quantity

    def peek(self) -> float:
        c = -1 if self.is_max_heap else 1

        if self.is_empty():
            return c*math.inf

        return c*self.heap[0][0]

    def peek_order(self) -> Order:
        if self.is_empty():
            return Order(price=0, agent_id=0, order_id=0, order_type=0, quantity=0, time=0)
        return self.heap[0][1]

    def clear(self):
        self.heap = []
        self.order_dict = {}
        self.deleted_ids = set()
        self.size = 0

    def is_empty(self) -> bool:
        return self.count() == 0

    def count(self) -> int:
        return self.size

    def remove(self, order_id: int):
        if not self.is_matched:
            if order_id in self.order_dict:
                self.deleted_ids.add(order_id)
                self.size -= self.order_dict[order_id].quantity
                del self.order_dict[order_id]
            if self.peek_order().order_id == order_id:
                heapq.heappop(self.heap)

    def contains(self, order_id: int) -> bool:
        return order_id in self.order_dict

    def push_to(self) -> Optional['Order']:
        while self.heap:
            price, order = heapq.heappop(self.heap)
            if order.order_id not in self.deleted_ids:
                self.size -= order.quantity
                del self.order_dict[order.order_id]

                # Make sure the new top of heap is not a removed order
                try:
                    while self.peek_order().order_id in self.deleted_ids:
                        heapq.heappop(self.heap)
                except IndexError:
                    pass

                return order
        return None

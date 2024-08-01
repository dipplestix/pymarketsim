import random
from marketsim.event.event_queue import EventQueue
from marketsim.fourheap.fourheap import FourHeap
from marketsim.fundamental.fundamental_abc import Fundamental
from marketsim.fourheap import constants


class Market:
    def __init__(self, fundamental: Fundamental, time_steps,  random_seed: int = None):
   
        random.seed(random_seed)

        self.order_book = FourHeap()
        self.matched_orders = []
        self.fundamental = fundamental
        self.event_queue = EventQueue(random_seed=random.randint(1,4096))
        self.end_time = time_steps


    def get_fundamental_value(self):
        t = self.get_time()
        return self.fundamental.get_value_at(t)

    def get_final_fundamental(self):
        return self.fundamental.get_final_fundamental()

    def withdraw_all(self, agent_id: int):
        self.order_book.withdraw_all(agent_id)

    def clear_market(self):
        new_orders = self.order_book.market_clear(self.get_time())
        self.matched_orders += new_orders
        return new_orders

    def add_orders(self, orders):
        for order in orders:
            self.event_queue.schedule_activity(order)

    def get_time(self):
        return self.event_queue.get_current_time()

    def get_info(self):
        return self.fundamental.get_info()

    def step(self):
        # TODO Need to figure out how to handle ties for price and time
        orders = self.event_queue.step()
        self.buy_init_volume, self.sell_init_volumn = 0, 0
        for order in orders:
            if order.quantity <= 0:
                continue
            self.order_book.insert(order)
        new_orders = self.clear_market()

        # Compute midprices.
        self.order_book.update_midprice()
        return new_orders

    def get_midprices(self):
        return self.order_book.midprices

    def reset(self, fundamental):
        self.order_book = FourHeap()
        self.matched_orders = []
        self.event_queue = EventQueue(random_seed=random.randint(1,4096))
        self.fundamental = fundamental

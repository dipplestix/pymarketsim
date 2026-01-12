import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List
import numpy as np


class ZIAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: List, pv_var: float, eta: float = 1.0):
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = PrivateValues(q_max, pv_var)
        self.position = 0
        self.shade = shade
        self.cash = 0
        self.eta = eta
        self._order_counter = 0  # Counter for unique order IDs (faster than random.randint)

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*val
        # print(f'It is time {t} with final time {T} and I observed {val} and my estimate is {rho, estimate}')
        return estimate

    def take_action(self, estimate=None):
        side = random.choice([BUY, SELL])
        t = self.market.get_time()
        if estimate is None:
            estimate = self.estimate_fundamental()
        spread = self.shade[1] - self.shade[0]
        valuation_offset = spread*random.random() + self.shade[0]

        # Cache private value lookup (avoid duplicate computation when eta != 1.0)
        pv_value = self.pv.value_for_exchange(self.position, side)

        if side == BUY:
            price = estimate + pv_value - valuation_offset
        else:
            price = estimate + pv_value + valuation_offset

        if self.eta != 1.0:
            base_price = estimate + pv_value
            if side == BUY:
                best_price = self.market.order_book.get_best_ask()
                if (base_price - best_price) > self.eta*valuation_offset and best_price != np.inf:
                    price = best_price
            else:
                best_price = self.market.order_book.get_best_bid()
                if (best_price - base_price) > self.eta*valuation_offset and best_price != np.inf:
                    price = best_price

        # Use counter for order ID (faster than random.randint)
        self._order_counter += 1
        order_id = self.agent_id * 1000000 + self._order_counter

        order = Order(
            price=price,
            quantity=1,
            agent_id=self.agent_id,
            time=t,
            order_type=side,
            order_id=order_id
        )

        return [order]

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def __str__(self):
        return f'ZI{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self):
        self.position = 0
        self.cash = 0
        self.pv = PrivateValues(self.q_max, self.pv_var)
        self._order_counter = 0


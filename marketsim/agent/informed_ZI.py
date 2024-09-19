import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List


class ZIAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: List, pv_var: float):
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = PrivateValues(q_max, pv_var)
        self.position = 0
        self.shade = shade
        self.cash = 0

    def get_id(self) -> int:
        return self.agent_id

    def take_action(self, side):
        t = self.market.get_time()
        # This line will make the agent informed.
        estimate = self.market.get_final_fundamental()
        spread = self.shade[1] - self.shade[0]
        valuation_offset = spread*random.random() + self.shade[0]
        if side == BUY:
            price = estimate + self.pv.value_for_exchange(self.position, BUY) - valuation_offset
        elif side == SELL:
            price = estimate + self.pv.value_for_exchange(self.position, SELL) + valuation_offset
        # print(f'It is time {t} and I am on {side}. My estimate is {estimate} and my marginal pv is '
        #       f'{self.pv.value_for_exchange(self.position, side)} with offset {valuation_offset}. '
        #       f'Therefore I offer price {price}')

        order = Order(
            price=price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=random.randint(1, 10000000)
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


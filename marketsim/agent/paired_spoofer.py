import random
import numpy as np
from agent.agent import Agent
from market.market import Market
from fourheap.order import Order
from private_values.private_values import PrivateValues
from fourheap.constants import BUY, SELL
from typing import List

'''
Functionally the same as a ZI agent. Mainly just a placeholder for paired instance exps.
'''
class SpooferZIAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: List, pv_var: float, pv = None):
        self.agent_id = agent_id
        self.market = market
        if pv != -1:
            self.pv = pv
        else:
            self.pv = PrivateValues(q_max, pv_var)
        self.position = 0
        self.shade = shade
        self.cash = 0

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

    def take_action(self, side, seed = None):
        t = self.market.get_time()
        random.seed(t + seed)
        spreadRand = random.random()
        orderId1 = random.randint(1, 10000000)
        orderId2 = random.randint(1, 10000000)
        
        # if 1000 < t < 1050:
        #     print(spreadRand, orderId1, orderId2)
        #     input()
        
        estimate = self.estimate_fundamental()
        spread = self.shade[1] - self.shade[0]
        valuation_offset = spread*spreadRand + self.shade[0]
        if side == BUY:
            price = estimate + self.pv.value_for_exchange(self.position, BUY) - valuation_offset
        elif side == SELL:
            price = estimate + self.pv.value_for_exchange(self.position, SELL) + valuation_offset
        if 1000 < t < 1500:
            print(f'It is time {t} and I am a spoofer. My side is {side}. My estimate is {self.estimate_fundamental()}, my position is {self.position}, my offset is {valuation_offset}, and my marginal pv is '
                f'{self.pv.value_for_exchange(self.position, SELL)}.'
                f'Therefore I offer price {price}')

        order = Order(
            price=price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=orderId1
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

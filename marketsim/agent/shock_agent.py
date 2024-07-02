import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List

import numpy as np
import math

class ShockAgent(Agent):
    def __init__(self, agent_id: int, market: Market, entry_time: int, shock_interval: int, shock_volume: int, side = SELL, random_seed: int = 0):
        
        if random_seed != 0:
            # torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.agent_id = agent_id
        self.market = market

        self.entry_time = entry_time
        self.exit_time = entry_time + shock_interval

        self.shock_interval = shock_interval
        self.shock_volume = shock_volume
        self.remaining_shock_volume = shock_volume

        self.side = side

        self.position = 0
        self.cash = 0
        

    def get_id(self) -> int:
        return self.agent_id

    def take_action(self, _):
        
        t = self.market.get_time()

        remaining_shock_time = self.shock_interval - (self.entry_time - t)

        max_volume = math.ceil(self.remaining_shock_volume / remaining_shock_time)
        
        orders = []

        if self.side == BUY:

            quantity = self.market.get_unmatched_sell_quantity()
            quantity = min(quantity, max_volume)

            if quantity != 0:
                orders.append(
                            Order(
                                price=np.inf,
                                quantity=quantity,
                                agent_id=self.get_id(),
                                time=t,
                                order_type=self.side,
                                order_id=random.randint(1, 10000000)
                                )
                            )
        
        elif self.side == SELL:

            quantity =  self.market.get_unmatched_buy_quantity()
            quantity = min(quantity, max_volume)

            if quantity != 0:
                orders.append(
                            Order(
                                price= -1*np.inf,
                                quantity=quantity,
                                agent_id=self.get_id(),
                                time=t,
                                order_type=self.side,
                                order_id=random.randint(1, 10000000)
                                )
                            )
        
        return orders

    def __str__(self):
        return f'ShockAgent_{self.agent_id}'
    
    def update_position(self, q, p):
        self.position += q
        self.remaining_shock_volume -= abs(q)
        self.cash += p

    # these are not relevant to shock agent
    def get_pos_value(self) -> float:
        return 0

    def reset(self):
        pass
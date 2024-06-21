import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List

import numpy as np

class ShockAgent(Agent):
    def __init__(self, agent_id: int, market: Market, entry_time: int, shock_interval: int, shock_volume: int, side = SELL):
        self.agent_id = agent_id
        self.market = market

        self.entry_time = entry_time
        self.exit_time = entry_time + shock_interval

        self.shock_interval = shock_interval
        self.shock_volume = shock_volume

        self.side = side

        self.position = 0
        self.cash = 0
        

    def get_id(self) -> int:
        return self.agent_id

    def take_action(self, side):
        assert side == BUY or side == SELL, "Side must be BUY or SELL"

        t = self.market.get_time()

        remaining_shock_time = self.shock_interval - (self.entry_time - t)

        max_volume = (self.shock_volume // remaining_shock_time)
        
        orders = []

        if side == BUY:
            quantity = 0
            for oid, order in self.market.order_book.sell_unmatched.order_dict.items():
                quantity += order.quantity

            quantity = min(quantity, max_volume)

            if quantity != 0:
                

                orders.append(
                    Order(
                        price=np.inf,
                        quantity=quantity,
                        agent_id=self.get_id(),
                        time=t,
                        order_type=side,
                        order_id=random.randint(1, 10000000)
                    )
                )

            # for oid, order in self.market.order_book.sell_unmatched.order_dict.items():
            #     # print(order)
            #     # print(order.price)
            #     price = order.price

            #     quantity = min(order.quantity, max_volume)

            #     max_volume -= quantity

            #     if quantity == 0:
            #         break

            #     orders.append(
            #         Order(
            #             price=price,
            #             quantity=quantity,
            #             agent_id=self.get_id(),
            #             time=t,
            #             order_type=side,
            #             order_id=random.randint(1, 10000000)
            #         )
            #     )
            
        elif side == SELL:

            quantity = 0
            for oid, order in self.market.order_book.buy_unmatched.order_dict.items():
                quantity += order.quantity

            quantity = min(quantity, max_volume)

            if quantity != 0:
                orders.append(
                    Order(
                        price=-1 * np.inf,
                        quantity=quantity,
                        agent_id=self.get_id(),
                        time=t,
                        order_type=side,
                        order_id=random.randint(1, 10000000)
                    )
                )

            # for oid, order in self.market.order_book.buy_unmatched.order_dict.items():
            #     # print(oid, order)
            #     # print(order.price)
            #     price = order.price
                
            #     quantity = min(order.quantity, max_volume)

            #     max_volume -= quantity

            #     if quantity == 0:
            #         break

            #     orders.append(
            #         Order(
            #             price=price,
            #             quantity=quantity,
            #             agent_id=self.get_id(),
            #             time=t,
            #             order_type=side,
            #             order_id=random.randint(1, 10000000)
            #         )
            #     )
        
        return orders

    def __str__(self):
        return f'ShockAgent_{self.agent_id}'


    # not sure these are relevant to shock agent
    def get_pos_value(self) -> float:
        return 0

    def reset(self):
        pass
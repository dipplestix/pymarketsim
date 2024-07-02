import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List

import numpy as np

class TrendAgent(Agent):
    def __init__(self, agent_id: int, market: Market, L: int, PI: float, random_seed: int = 0):
        
        if random_seed != 0:
            # torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.agent_id = agent_id
        self.market = market
        self.L = L
        self.PI = PI
        
        self.position = 0
        self.cash = 0

    def get_id(self) -> int:
        return self.agent_id

    def take_action(self, _):
        
        # print("TREND")
        past_transactions = self.market.transaction_prices

        # print(past_transactions)

        if len(past_transactions) < self.L:
            return []
        
       
        past_transactions = past_transactions[-self.L:]
        # print(past_transactions)
        t = self.market.get_time()        
        orders = []
        
        # print(past_transactions)
        # print(len(past_transactions), self.L)

        previous_up = past_transactions[0] - 1
        previous_down = past_transactions[0] + 1

        increasing = True
        decreasing = True

        for price in past_transactions:
            if previous_up >= price:
                increasing = False

            if previous_down <= price:
                decreasing = False

        # if increasing or decreasing:
        #     print(f"Results: {t} {increasing} {decreasing}")

        if increasing:
            
            price = self.market.order_book.get_best_ask()

            next_lowest = self.market.order_book.get_next_best_ask()

            price = min(price + self.PI, max(price, next_lowest - 1))

            if price != np.inf and price != -1 * np.inf:
                # accepts outstanding offer
                orders.append(
                    Order(
                    price=self.market.order_book.get_best_ask(),
                    quantity=1,
                    agent_id=self.get_id(),
                    time=t,
                    order_type=BUY,
                    order_id=random.randint(1, 10000000)
                    )
                )

                # places trend offer
                orders.append(
                    Order(
                    price=price,
                    quantity=1,
                    agent_id=self.get_id(),
                    time=t,
                    order_type=SELL,
                    order_id=random.randint(1, 10000000)
                    )
                )
        elif decreasing:
            price = self.market.order_book.get_best_ask()
            
            next_highest= self.market.order_book.get_next_best_bid()



            price = max(price - self.PI, min(price, next_highest + 1))

            if price != np.inf and price != -1 * np.inf:
                # accepts outstanding offer
                orders.append(
                    Order(
                    price=self.market.order_book.get_best_bid(),
                    quantity=1,
                    agent_id=self.get_id(),
                    time=t,
                    order_type=BUY,
                    order_id=random.randint(1, 10000000)
                    )
                )

                # places trend offer
                orders.append(
                    Order(
                    price=price,
                    quantity=1,
                    agent_id=self.get_id(),
                    time=t,
                    order_type=SELL,
                    order_id=random.randint(1, 10000000)
                    )
                )

        return orders 

    def __str__(self):
        return f'TrendAgent{self.agent_id}'

    # not sure these are relevant to shock agent
    def get_pos_value(self) -> float:
        return 0.0

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def reset(self):
        self.position = 0
        self.cash  = 0

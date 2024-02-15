import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List


class MMAgent(Agent):
    def __init__(self, agent_id: int, market: Market, xi: float, K: int, omega: float):
        self.agent_id = agent_id
        self.market = market

        self.position = 0
        self.cash = 0

        self.xi = xi
        self.K = K
        self.omega = omega

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*val
        return estimate

    def take_action(self):
        t = self.market.get_time()
        estimate = self.estimate_fundamental()
        orders = []
        st = estimate + 1/2*self.omega
        bt = estimate - 1 / 2 * self.omega

        for k in range(self.K):
            orders.append(
                Order(
                    price= bt - (k + 1)*self.xi,
                    quantity=1,
                    agent_id=self.get_id(),
                    time=t,
                    order_type=BUY,
                    order_id=random.randint(1, 10000000)
                )
            )
            orders.append(
                Order(
                    price= st + (k + 1)*self.xi,
                    quantity=1,
                    agent_id=self.get_id(),
                    time=t,
                    order_type=SELL,
                    order_id=random.randint(1, 10000000)
                )
            )

        return orders

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def __str__(self):
        return f'MM{self.agent_id}'

import random
import numpy as np
import scipy
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.fourheap.constants import BUY, SELL
from tianshou.policy.base import BasePolicy
from typing import List, Optional

"""
This market maker applies the beta policy, which generalizes multiple market making strategies.
https://arxiv.org/abs/2207.03352
"""

def ScaledBetaDist(x, n_levels, a, b):
    dist = scipy.stats.beta(a, b)
    return 1 / n_levels * dist.cdf(x / n_levels)


def quantise_scaledbetadist(total_volume, n_levels, a, b):
    probs = []
    for i in range(n_levels):
        prob = ScaledBetaDist(i + 1, n_levels, a, b) - ScaledBetaDist(i, n_levels, a, b)
        probs.append(prob)

    probs = np.array(probs) / np.sum(probs)
    order_profile = np.round(probs * total_volume)

    return order_profile


class MMAgent(Agent):
    def __init__(self, agent_id: int,
                 market: Market,
                 n_levels: int,
                 total_volume: int,
                 xi: float,
                 omega: float,
                 beta_params: dict=None,
                 policy: Optional[BasePolicy]=None):

        self.agent_id = agent_id
        self.market = market

        self.position = 0
        self.cash = 0

        self.n_levels = n_levels
        self.beta_params = beta_params
        self.policy = policy
        self.total_volume = total_volume

        self.xi = xi
        self.omega = omega


    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1 - r) ** (T - t)

        estimate = (1 - rho) * mean + rho * val
        return estimate

    def take_action(self):
        t = self.market.get_time()
        orders = []

        if self.policy is not None:
            # Get MM obs and apply the policy.
            pass
        else:
            a_buy = self.beta_params['a_buy']
            b_buy = self.beta_params['b_buy']
            a_sell = self.beta_params['a_sell']
            b_sell = self.beta_params['b_sell']

        buy_orders = quantise_scaledbetadist(total_volume=self.total_volume,
                                             n_levels=self.n_levels,
                                             a=a_buy,
                                             b=b_buy)

        sell_orders = quantise_scaledbetadist(total_volume=self.total_volume,
                                             n_levels=self.n_levels,
                                             a=a_sell,
                                             b=b_sell)

        # Get the best bid and best ask
        best_ask = self.market.order_book.get_ask_quote()
        best_bid = self.market.order_book.get_bid_quote()

        estimate = self.estimate_fundamental()
        st = max(estimate + 1 / 2 * self.omega, best_bid)
        bt = min(estimate - 1 / 2 * self.omega, best_ask)

        for k in range(self.n_levels):
            orders.append(
                Order(
                    price= bt - (k + 1) * self.xi,
                    quantity=buy_orders[k],
                    agent_id=self.get_id(),
                    time=t,
                    order_type=BUY,
                    order_id=random.randint(1, 10000000)
                )
            )

            orders.append(
                Order(
                    price=st + (k + 1) * self.xi,
                    quantity=sell_orders[k],
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

    def update_policy(self, new_policy):
        self.policy = new_policy

    def update_beta_params(self, new_beta_params):
        self.beta_params = new_beta_params

    def __str__(self):
        return f'MM{self.agent_id}'

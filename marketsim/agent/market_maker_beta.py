import random
import numpy as np
import scipy
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.fourheap.constants import BUY, SELL
from typing import List, Optional, Any

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
    def __init__(self,
                 agent_id: int,
                 market: Market,
                 n_levels: int,
                 total_volume: int,
                 xi: float,
                 omega: float,
                 beta_params: dict=None,
                 policy: bool=False,
                 inv_driven=False,
                 w0=5,
                 p=2,
                 k_min=5,
                 k_max=20,
                 max_position=15):

        self.agent_id = agent_id
        self.market = market

        self.position = 0
        self.cash = 0

        self.n_levels = n_levels
        self.beta_params = beta_params
        self.policy = policy
        self.inv_driven = inv_driven
        self.total_volume = total_volume

        self.xi = xi
        self.omega = omega

        self.last_value = 0

        # Inventory-driven policy
        self.w0 = w0
        self.p = p
        self.k_max = k_max
        self.k_min = k_min
        self.max_position = max_position


    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1 - r) ** (T - t)

        estimate = (1 - rho) * mean + rho * val
        return estimate

    def take_action(self, action=None):
        t = self.market.get_time()
        orders = []

        if self.policy:
            # RL policy
            a_buy, b_buy, a_sell, b_sell = action
        elif self.inv_driven:
            # Inventory driven policy
            a_buy, b_buy, a_sell, b_sell = self.inv_driven_policy()
        else:
            # Static beta policy
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
        best_ask = self.market.order_book.get_best_ask()
        best_bid = self.market.order_book.get_best_bid()

        estimate = self.estimate_fundamental()
        st = max(estimate + 1 / 2 * self.omega, best_bid)
        bt = min(estimate - 1 / 2 * self.omega, best_ask)

        # TODO: Is normalizer needed?
        for k in range(self.n_levels):
            orders.append(
                Order(
                    price= bt - (k + 1) * self.xi,
                    quantity=int(buy_orders[k]),
                    agent_id=self.get_id(),
                    time=t,
                    order_type=BUY,
                    order_id=random.randint(1, 10000000)
                )
            )

            orders.append(
                Order(
                    price=st + (k + 1) * self.xi,
                    quantity=int(sell_orders[k]),
                    agent_id=self.get_id(),
                    time=t,
                    order_type=SELL,
                    order_id=random.randint(1, 10000000)
                )
            )

        # print("orders:", orders)
        return orders

    def inv_driven_policy(self):
        clamp = min(-1, abs(self.position / self.max_position)) ** self.p
        f1 = 1 / self.w0 * (1 + (1 / self.w0 - 1) * clamp)
        f2 = 1 / self.w0 * (1 - clamp)
        if self.position >= 0:
            w_bid = f1
            w_ask = f2
        else:
            w_bid = f2
            w_ask = f1

        k = (self.k_max - self.k_min) * clamp + self.k_min

        a_buy = w_bid * (k - 2) + 1
        b_buy = (1 - w_bid) * (k - 2) + 1
        a_sell = w_ask * (k - 2) + 1
        b_sell =(1 - w_ask) * (k - 2) + 1

        return a_buy, b_buy, a_sell, b_sell


    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def update_beta_params(self, new_beta_params):
        self.beta_params = new_beta_params

    def __str__(self):
        return f'MM{self.agent_id}'

    def reset(self):
        self.position = 0
        self.cash = 0
        self.last_value = 0

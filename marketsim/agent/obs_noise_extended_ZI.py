import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List


class ZIAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, offset: float, eta: float, shade: List, obs_var: int):
        self.agent_id = agent_id
        self.market = market
        self.pv = PrivateValues(q_max)
        self.position = 0
        self.offset = offset
        self.eta = eta
        self.shade = shade
        self.obs_var = obs_var
        self.obs_std = obs_var ** 0.5
        self.cash = 0

        self.prior_mean = 0;
        self.prior_time = 0
        self.prior_var = 0

    def update_priors(self, time):
        
        var_tilda = (time - self.prior_time)

        self.prior_time = time



    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        noisy_val = self.market.get_fundamental_value() + random.gauss(0, self.obs_std)

        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*noisy_val
        return estimate

    def take_action(self, side):
        assert side == BUY or side == SELL, "Side must be BUY or SELL"
        t = self.market.get_time()
        estimate = self.estimate_fundamental()
        spread = self.shade[1] - self.shade[0]
        valuation_offset = spread*random.random() + self.shade[0]
        if side == BUY:
            price = estimate + self.pv.value_for_exchange(self.position, BUY) - valuation_offset
        else:
            price = estimate + self.pv.value_for_exchange(self.position, SELL) + valuation_offset
        # print(f'It is time {t} and I am on {side}. My estimate is {estimate} and my marginal pv is '
        #       f'{self.pv.value_for_exchange(self.position, side)} with offset {valuation_offset}. '
        #       f'Therefore I offer price {price}')

        if self.eta != 1.0:
            surplus = valuation_offset
            if side == BUY:
                best_price = self.market.order_book.get_best_ask()
                if (best_price - price) > self.eta*surplus:
                    price = best_price
            else:
                best_price = self.market.order_book.get_best_ask()
                if (price - best_price) > self.eta*surplus:
                    price = best_price

        order = Order(
            price=price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=random.randint(1, 10000000)
        )
        # print(f'It is timestep {t} and I am assigned {side}. I am making an order with price {price} since my estimate is {estimate} ')
        # print(f'My current position is {self.position} and my private values are {self.pv.values}')

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
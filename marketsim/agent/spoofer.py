import random
import math
from agent.agent import Agent
from market.market import Market
from fourheap.order import Order
from private_values.private_values import PrivateValues
from fourheap.constants import BUY, SELL


class SpoofingAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, pv_var: float, order_size:int, spoofing_size: int, normalizers: dict, learning:bool):
        self.agent_id = agent_id
        self.market = market
        self.pv = PrivateValues(q_max, pv_var)
        self.position = 0
        self.spoofing_size = spoofing_size
        self.order_size = order_size
        self.cash = 0
        self.last_value = 0 # value at last time step (liquidate all inventory)
        self.normalizers = normalizers # A dictionary {"fundamental": float, "invt": float, "cash": float}
        self.position_track = []
        self.learning = learning

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho) * mean + rho*val
        # print(f'It is time {t} with final time {T} and I observed {val} and my estimate is {rho, estimate}')
        return estimate

    def take_action(self, action):
        '''
            action: tuple (offset from price quote, offset from valuation)
        '''
        t = self.market.get_time()
        regular_order_offset, spoofing_order_offset = action
        # Normalization constants need to be tuned
        if self.learning:
            unnormalized_reg_offset = regular_order_offset * 4500
            unnormalized_spoof_offset = spoofing_order_offset * 200
        else:
            #Check with Xintong
            unnormalized_reg_offset = 10
            unnormalized_spoof_offset = 1
        if math.isinf(self.market.order_book.buy_unmatched.peek()):
            # Should rarely happen since the spoofer enters after t = 1000
            # If it does, just submit a bid that won't lose the spoofer money
            spoofing_price = self.estimate_fundamental() + self.pv.value_for_exchange(self.position, BUY)
        else:
            spoofing_price = self.market.order_book.buy_unmatched.peek() - unnormalized_spoof_offset
        
        regular_order_price = self.estimate_fundamental() + self.pv.value_for_exchange(self.position, SELL) + unnormalized_reg_offset
        orders = []

        # Regular order.
        regular_order = Order(
            price=regular_order_price,    
            quantity=self.order_size,
            agent_id=self.get_id(),
            time=t,
            order_type=SELL,
            order_id=random.randint(1, 10000000)
        )
        orders.append(regular_order)
        
        # Spoofing Order
        spoofing_order = Order(
            price=spoofing_price,
            quantity=self.spoofing_size,
            agent_id=self.get_id(),
            time=t,
            order_type=BUY,
            order_id=random.randint(1, 10000000)
        )
        orders.append(spoofing_order)
        # input(orders)
        return orders

    def update_position(self, q, p):
        self.position += q
        self.cash += p
        self.position_track.append(self.position)

    def __str__(self):
        return f'SPF{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self):
        self.position = 0
        self.cash = 0
        self.last_value = 0



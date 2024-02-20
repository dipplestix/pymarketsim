import random
import math
from scipy.interpolate import CubicSpline
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List

#TODO: Import the lam variable somehow
GRACE_PERIOD = 1/0.5


class HBLAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, offset: float, shade: List, L:int):
        self.agent_id = agent_id
        self.market = market
        self.pv = PrivateValues(q_max)
        self.position = 0
        self.offset = offset
        self.shade = shade
        self.cash = 0
        self.L = L

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*val
        return estimate

    def get_last_trade_time_step(self):
        '''
            Gets time step boundary for orders occurring since the earliest 
            order contributing to the Lth most recent trade up to most recent trade time.
        '''
        #Assumes that matched_orders is ordered by timestep of trades
        last_matched_order_ind = len(self.market.matched_orders) - self.L*2
        if len(self.market.matched_orders) < 2:
            most_recent_trade = self.market.get_time()
        else:
            most_recent_trade = max(self.market.matched_orders[-1].order.time, self.market.matched_orders[-2].order.time)
        #Returns earliest contributing buy/sell
        lth_trade = min(self.market.matched_orders[last_matched_order_ind].order.time, self.market.matched_orders[last_matched_order_ind + 1].order.time)
        return lth_trade, most_recent_trade

    def belief_function(self, p, side, orders):
        '''
            Defined over all past bid/ask prices. Need cubic spline interpolation for prices not encountered previously.
            Returns: probability that bid/ask will execute at price p.
        '''
        t = self.market.get_time()
        if side == BUY:
            TBL = 0 #Transact bids less or equal
            AL = 0 #Asks less or equal
            RBG = 0 #Rejected bids greater or equal
            for matched_order in self.market.matched_orders:
                if matched_order.order.order_type == BUY and matched_order.price <= p:    
                    TBL += 1 
            for order in orders:
                if order.price <= p and order.order_type == SELL:
                    AL += 1
                #TODO: FIX NUANCES TO DO WITH REJECTED TRADES
                already_matched = False
                if order.order_type == BUY and order.price >= p:
                    for matched_order in self.market.matched_orders:
                        if order.order_id == matched_order.order.order_id:
                            #TODO: Account for case of matched order alive period > grace period
                            already_matched = True
                            break
                    if not already_matched and t - order.time > GRACE_PERIOD:
                        RBG += 1
            return (TBL + AL) / (TBL + AL + RBG)
        else:
            TAG = 0  #Transact ask greater or equal
            BG = 0 #Bid greater or equal
            RAL = 0 #Reject ask less or equal
            for matched_order in self.market.matched_orders:
                if matched_order.order.order_type == SELL and matched_order.price >= p:    
                    TAG += 1 
            for order in orders:
                if order.price >= p and order.order_type == BUY:
                    BG += 1
                #TODO: FIX NUANCES TO DO WITH REJECTED TRADES
                already_matched = False
                if order.order_type == SELL and order.price <= p:
                    for matched_order in self.market.matched_orders:
                        if order.order_id == matched_order.order.order_id:
                            already_matched = True
                            break
                    if not already_matched:
                        print(order, t)
                    if not already_matched and t - order.time > GRACE_PERIOD:
                        RAL += 1
            # if (p == 3.2):
            #     print(TAG, BG, RAL)
            return (TAG + BG) / (TAG + BG + RAL)

    def belief_interpolation(self, side):
        last_L_orders = []
        lower_bound_mem, upper_bound_mem = self.get_last_trade_time_step()
        print(lower_bound_mem, upper_bound_mem)
        print("_______________")
        #Loop for orders from earliest contributing time step to current (inclusive) 
        for time in range(lower_bound_mem, upper_bound_mem + 1):
                for order in self.market.event_queue.scheduled_activities[time]:
                    last_L_orders.append(order)            
        last_L_side_prices = []
        for order in last_L_orders:
            if order.order_type == side:
                last_L_side_prices.append(order.price)
        last_L_side_prices.sort()
        last_L_belief = [self.belief_function(price, side, last_L_orders) for price in last_L_side_prices]
        return last_L_side_prices, CubicSpline(last_L_side_prices, last_L_belief)

    def determine_optimal_price(self, side):
        '''
            Reference: https://www.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2009/notes/joel.pdf
            http://spider.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2005/notes/das.pdf 
        '''
        #TODO: Fix this. Belief interpolation only returns 2 values for debugging.
        _, interpolated_belief = self.belief_interpolation(side)
        if self.market.order_book.sell_unmatched.peek_order():
            outstanding_ask = self.market.order_book.sell_unmatched.peek_order().price #Lowest ask up to current timestep
        else:
            outstanding_ask = self.estimate_fundamental() + 10
        
        if self.market.order_book.buy_unmatched.peek_order():
            outstanding_bid = self.market.order_book.buy_unmatched.peek_order().price #Highest bid up to curent timestep
        else:
            outstanding_bid = self.estimate_fundamental() - 10
        
        max_eval_price = 0
        opt_price = 0
        if side == BUY:
            #Range should be (ob, oa] for buy
            outstanding_bid += 1
            outstanding_ask += 1
        print(outstanding_ask, outstanding_bid)
        #For sell, range is [ob, oa) --> Assume integer price ticks
        #TODO: math.floor is temporary
        for price in range(math.floor(outstanding_bid), math.ceil(outstanding_ask)):
            if side == BUY:
                eval_price = (self.estimate_fundamental() + self.pv.value_at_position(self.position + 1) - price) * interpolated_belief(price)
            else:
                #Sell
                eval_price = (price - self.estimate_fundamental() - self.pv.value_at_position(self.position)) * interpolated_belief(price)
            if eval_price > max_eval_price:
                max_eval_price = eval_price
                opt_price = price
        return opt_price

    def take_action(self, side):
        '''
            Behavior reverts to ZI agent if L > total num of trades executed.
        '''
        t = self.market.get_time()
        estimate = self.estimate_fundamental()
        spread = self.shade[1] - self.shade[0]
        if len(self.market.matched_orders) >= self.L:
            opt_price = self.determine_optimal_price(side)
            return Order(
                price=opt_price,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=side,
                order_id=random.randint(1, 10000000)
                        )
        else:
            if side == BUY:
                price = estimate + self.pv.value_at_position(self.position + 1) + side*spread*random.random() + self.shade[0]
            else:
                price = estimate + self.pv.value_at_position(self.position) + side*spread*random.random() + self.shade[0]
            return Order(
                price=price,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=side,
                order_id=random.randint(1, 10000000)
                        )

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def __str__(self):
        return f'ZI{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

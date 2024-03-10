import random
import math
import sys
import scipy as sp
import matplotlib.pyplot as plt
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List

# TODO: Import the lam variable somehow
GRACE_PERIOD = 1/0.5
EPSILON = 1e-5


class HBLAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, offset: float, shade: List, L: int):
        self.agent_id = agent_id
        self.market = market
        self.pv = PrivateValues(q_max)
        self.position = 0
        self.offset = offset
        self.shade = shade
        self.cash = 0
        self.L = L
        self.ORDERS = 0
        self.HBL_MOVES = 0

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
        # Assumes that matched_orders is ordered by timestep of trades
        last_matched_order_ind = len(self.market.matched_orders) - self.L*2
        if len(self.market.matched_orders) < 2:
            most_recent_trade = self.market.get_time()
        else:
            most_recent_trade = max(
                self.market.matched_orders[-1].order.time, self.market.matched_orders[-2].order.time)
        # Gets earliest contributing buy/sell
        lth_trade = min(self.market.matched_orders[last_matched_order_ind].order.time,
                        self.market.matched_orders[last_matched_order_ind + 1].order.time)
        return lth_trade, most_recent_trade

    def belief_function(self, p, side, orders, lower_bound_time, upper_bound_time):
        '''
            Defined over all past bid/ask prices in orders. 
            Need cubic spline interpolation for prices not encountered previously.
            Returns: probability that bid/ask will execute at price p.
            TODO: Optimize? And optimize for history changes as orders are added
        '''
        current_time = self.market.get_time()
        if side == BUY:
            TBL = 0  # Transact bids less or equal
            AL = 0  # Asks less or equal
            RBG = 0  # Rejected bids greater or equal
            for order in orders:
                if order.price <= p and order.order_type == SELL:
                    AL += 1

            for order in orders:
                found_matched = False
                for matched_order in self.market.matched_orders:
                    if order.order_id == matched_order.order.order_id:
                        if matched_order.order.order_type == BUY and matched_order.order.price <= p:
                            TBL += 1
                        alive_period = matched_order.time - order.time
                        found_matched = True
                if found_matched:
                    if alive_period > GRACE_PERIOD:
                        RBG += 1
                else:
                    if current_time - order.time > GRACE_PERIOD:
                        RBG += 1
                    else:
                        RBG += (current_time -
                                order.time) / GRACE_PERIOD
            if TBL + AL == 0:
                return 0
            else:
                return (TBL + AL) / (TBL + AL + RBG)

        else:
            TAG = 0  # Transact ask greater or equal
            BG = 0  # Bid greater or equal
            RAL = 0  # Reject ask less or equal

            for order in orders:
                if order.price >= p and order.order_type == BUY:
                    BG += 1

            # TODO: Withdraw order?
            for order in orders:
                found_matched = False
                for matched_order in self.market.matched_orders:
                    if order.order_id == matched_order.order.order_id:
                        if matched_order.order.order_type == SELL and matched_order.order.price >= p:
                            TAG += 1
                        alive_period = matched_order.time - order.time
                        found_matched = True
                        break
                if found_matched:
                    if alive_period > GRACE_PERIOD:
                        RAL += 1
                else:
                    if current_time - order.time >= GRACE_PERIOD:
                        RAL += 1
                    else:
                        RAL += (current_time -
                                order.time) / GRACE_PERIOD
            if TAG + BG == 0:
                return 0
            else:
                return (TAG + BG) / (TAG + BG + RAL)

    def determine_optimal_price(self, side):
        '''
            Reference: https://www.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2009/notes/joel.pdf
            http://spider.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2005/notes/das.pdf 
        '''

        lower_bound_mem, upper_bound_mem = self.get_last_trade_time_step()

        last_L_orders = []
        for time in range(lower_bound_mem, upper_bound_mem + 1):
            for order in self.market.event_queue.scheduled_activities[time]:
                last_L_orders.append(order)

        best_sell_price = float(self.market.order_book.sell_unmatched.peek())
        best_buy_price = float(self.market.order_book.buy_unmatched.peek())
        if side == BUY:
            lowest_buy_price = -float(self.market.order_book.buy_unmatched.heap[-1][0])
            mid_buy_price = (best_buy_price + lowest_buy_price) / 2
            # Interpolate best ask - best buy - mid buy - lowest buy/0
            best_buy_belief = self.belief_function(
                best_buy_price, BUY, last_L_orders, lower_bound_mem, upper_bound_mem)
            mid_buy_belief = self.belief_function(
                mid_buy_price, BUY, last_L_orders, lower_bound_mem, upper_bound_mem)
            lowest_buy_belief = self.belief_function(
                lowest_buy_price, BUY, last_L_orders, lower_bound_mem, upper_bound_mem)
            if lowest_buy_belief > EPSILON:
                lowest_buy_price = 0
                lowest_buy_belief = 0

            prices = list(dict.fromkeys(
                [lowest_buy_price, mid_buy_price, best_buy_price, best_sell_price]))
            beliefs = [lowest_buy_belief, mid_buy_belief, best_buy_belief, 1]
            beliefs = beliefs[len(beliefs) - len(prices):]
            try:
                cs = sp.interpolate.CubicSpline(
                    prices, beliefs, extrapolate=True)
            except:
                print(self.market.order_book.buy_unmatched.heap)
                raise Exception("exception", prices, beliefs)

            def optimize(price): return (self.estimate_fundamental(
            ) + self.pv.value_at_position(self.position + 1) - price) * cs(price)
            try:
                max_x = sp.optimize.minimize_scalar(optimize, bounds=(
                    float(prices[0]), float(prices[-1])), method='bounded')
            except:
                raise Exception("e", prices, beliefs)
        else:
            # Price of highest sell
            highest_sell_price = float(self.market.order_book.sell_unmatched.heap[-1][0])
            mid_sell_price = (best_sell_price + highest_sell_price) / 2
            # Interpolate best ask - best buy - mid buy - lowest buy/0
            best_sell_belief = self.belief_function(
                best_sell_price, SELL, last_L_orders, lower_bound_mem, upper_bound_mem)
            mid_sell_belief = self.belief_function(
                mid_sell_price, SELL, last_L_orders, lower_bound_mem, upper_bound_mem)
            highest_sell_belief = self.belief_function(
                highest_sell_price, SELL, last_L_orders, lower_bound_mem, upper_bound_mem)
            # Option 1: If highest sell belief >> 0, set highest belief to sys.maxsize and then interpolate that way.
            # If we do this, then interpolate in segments or else the scale of maxsize messes up the overall interpolation
            # if highest_sell_belief > EPSILON:
            #     highest_sell_price = sys.maxsize
            #     highest_sell_belief = 0

            # Option 2: Use the interpolation up to the highest sell and then just extrapolate to the price point that has belief ~ 0.
            prices = list(dict.fromkeys(
                [best_buy_price, best_sell_price, mid_sell_price, highest_sell_price]))
            # If prices truncates, beliefs takes first n candidate values. Mid only = best or highest when 1 order in book
            beliefs = [1, best_sell_belief, mid_sell_belief,
                       highest_sell_belief][:len(prices)]
            try:
                cs = sp.interpolate.CubicSpline(prices, beliefs)
            except:
                raise Exception("exception SELL", prices, beliefs)
            # Assuming there's only 1 root which should be the case.
            # TODO: Account for case where no roots?
            if beliefs[-1] != 0 and len(cs.roots()) > 0:
                max_price = cs.roots()[-1]
                if max_price < prices[-1]:
                    max_price = sys.maxsize
            else:
                max_price = sys.maxsize
            prices.append(max_price)
            beliefs.append(0)

            def optimize(price): return (
                price - self.pv.value_at_position(self.position) - self.estimate_fundamental()) * cs(price)

            try:
                max_x = sp.optimize.minimize_scalar(optimize, bounds=(
                    float(prices[0]), float(prices[-1])), method='bounded')
            except:
                print(cs.roots(extrapolate=True))
                fig, ax = plt.subplots()
                ax.plot(prices, cs(prices))
                print(prices, beliefs)
                plt.show()
                raise Exception("e", prices, beliefs)

        return max_x.x.item()

    def take_action(self, side):
        '''
            Behavior reverts to ZI agent if L > total num of trades executed.
        '''
        t = self.market.get_time()
        estimate = self.estimate_fundamental()
        spread = self.shade[1] - self.shade[0]
        self.ORDERS = self.ORDERS + 1
        # print("BUY")
        # print(self.market.order_book.buy_unmatched.is_empty())
        # input(len(self.market.order_book.buy_unmatched.heap))
        # print("SELl")
        # print(self.market.order_book.sell_unmatched.is_empty())
        # input(len(self.market.order_book.sell_unmatched.heap))

        if len(self.market.matched_orders) >= 2 * self.L and not self.market.order_book.buy_unmatched.is_empty() and not self.market.order_book.sell_unmatched.is_empty():
            opt_price = self.determine_optimal_price(side)
            self.HBL_MOVES = self.HBL_MOVES + 1
            return Order(
                price=opt_price,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=side,
                order_id=random.randint(1, 10000000)
            )
        else:
            # ZI Agent
            if side == BUY:
                price = estimate + self.pv.value_at_position(
                    self.position + 1) + side*spread*random.random() + self.shade[0]
            else:
                price = estimate + self.pv.value_at_position(
                    self.position) + side*spread*random.random() + self.shade[0]
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

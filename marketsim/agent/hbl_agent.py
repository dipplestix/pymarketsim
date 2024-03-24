import random
import sys
import scipy as sp
import time
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
        return lth_trade, self.market.get_time()

    def belief_function(self, p, side, orders):

        '''
            Defined over all past bid/ask prices in orders. 
            Need cubic spline interpolation for prices not encountered previously.
            Returns: probability that bid/ask will execute at price p.
            TODO: Optimize? And optimize for history changes as orders are added
        '''
        #start_time = t.time()
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
                        found_matched = True
                if not found_matched:
                    if order.order_type == BUY and order.price >= p: 
                        if current_time - order.time > GRACE_PERIOD:
                            RBG += 1
                        else:
                            RBG += (current_time -
                                order.time) / GRACE_PERIOD
            #print("BUY BELIEF TIME", t.time() - start_time)
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
                        found_matched = True
                        break
                if not found_matched:
                    if order.order_type == SELL and order.price <= p: 
                        if current_time - order.time >= GRACE_PERIOD:
                            RAL += 1
                        else:
                            RAL += (current_time -
                                order.time) / GRACE_PERIOD
            #print("SELL BELIEF TIME", t.time() - start_time)
            if TAG + BG == 0:
                return 0
            else:
                return (TAG + BG) / (TAG + BG + RAL)

    def determine_optimal_price(self, side):
        '''
            Reference: https://www.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2009/notes/joel.pdf
            http://spider.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2005/notes/das.pdf 

            Spline time is ~97% of the run time. 0.3345/0.346
        '''
        #start_time = t.time()
        lower_bound_mem, upper_bound_mem = self.get_last_trade_time_step()

        last_L_orders = []
        for time in range(lower_bound_mem, upper_bound_mem + 1):
            for order in self.market.event_queue.scheduled_activities[time]:
                last_L_orders.append(order)

        # best_sell_price = float(self.market.order_book.sell_unmatched.peek())
        # best_buy_price = float(self.market.order_book.buy_unmatched.peek())
        # Not sure if it should be highest buy price in memory or highest buy price in LOB
        lowest_buy_price = sys.maxsize
        lowest_sell_price = sys.maxsize
        highest_buy_price = 0
        highest_sell_price = 0
        
        for order in last_L_orders:
            if order.order_type == BUY:
                lowest_buy_price = min(lowest_buy_price, order.price)
                highest_buy_price = max(highest_buy_price, order.price)
            else:
                lowest_sell_price = min(lowest_sell_price, order.price)
                highest_sell_price = max(highest_sell_price, order.price)
        lowest_buy_price = float(lowest_buy_price)
        highest_buy_price = float(highest_buy_price)

        lowest_sell_price = float(lowest_sell_price)
        highest_sell_price = float(highest_sell_price)

        if side == BUY:
            # print(self.market.order_book.buy_unmatched.heap)
            # print(lowest_buy_price)
            # input(self.market.order_book.buy_unmatched)
            mid_buy_price = (lowest_buy_price + highest_buy_price) / 2
            # Interpolate best ask - best buy - mid buy - lowest buy/0
            #start_spline_time = t.time()
            prices = []
            for order in last_L_orders:
                if order.order_type == BUY:
                    prices.append(float(order.price))
            prices = sorted(list(dict.fromkeys(prices)))
            beliefs = [self.belief_function(price, BUY, last_L_orders) for price in prices]
            
            # print(prices)
            # input(beliefs)

            # beliefs[-1] = 1
            # if beliefs[0] != 0:
            #     prices.insert(0, 0)
            #     beliefs.insert(0, 0)
            try:
                cs = sp.interpolate.CubicSpline(prices, beliefs, extrapolate=True)
            except:
                print("BUY")
                print(prices, beliefs)
                input()
            #print("SPLINE TIME BUY", t.time() - start_spline_time)
            # print(beliefs)
            # input(prices)

            def optimize(price): return -((self.estimate_fundamental(
            ) + self.pv.value_at_position(self.position + 1) - price) * self.belief_function(price, BUY, last_L_orders))

            #start_optimize_time = t.time()
            max_x = sp.optimize.minimize_scalar(optimize, bounds=(
                float(prices[0]), float(prices[-1])), method='bounded')
            #print("BUY OPTIMIZE", t.time() - start_optimize_time)
            #print("BUY_TIME", t.time() - start_time)
            return max_x.x.item()
        else:

            # print(highest_sell_price)
            # print(self.market.order_book.sell_unmatched)
            # input(self.market.order_book.sell_unmatched.heap)
            mid_sell_price = (lowest_sell_price + highest_sell_price) / 2
            # Interpolate best ask - best buy - mid buy - lowest buy/0
 
            # Option 1: If highest sell belief >> 0, set highest belief to sys.maxsize and then interpolate that way.
            # If we do this, then interpolate in segments or else the scale of maxsize messes up the overall interpolation
            # if highest_sell_belief > EPSILON:
            #     highest_sell_price = sys.maxsize
            #     highest_sell_belief = 0

            # Option 2: Use the interpolation up to the highest sell and then just extrapolate to the price point that has belief ~ 0.
            prices = []
            for order in last_L_orders:
                if order.order_type == SELL:
                    prices.append(float(order.price))
            prices = sorted(list(dict.fromkeys(prices)))
            beliefs = [self.belief_function(price, SELL, last_L_orders) for price in prices]
            #start_spline_time = t.time()
            # print("SELL")
            # print(prices)
            # input(beliefs)
            #start_optimize_time = t.time()
            # try:
            #     cs = sp.interpolate.CubicSpline(prices, beliefs, extrapolate=True)
            # except:
            #     print("SELL")
            #     print(prices, beliefs)
            #     print(self.market.order_book.sell_unmatched.heap[-1][0])
            #     input()
            def optimize(price): return -((
                price - self.pv.value_at_position(self.position) - self.estimate_fundamental()) * self.belief_function(price, SELL, last_L_orders))
            max_x = sp.optimize.minimize_scalar(optimize, bounds=(
                    float(prices[0]), prices[-1]), method='bounded')
            # if beliefs[-1] > 0:
            #     # 3 options: 1. Extrapolate roots. If doesn't work because graph turns up after final point
            #     # 2. interpolate linearly based on descent from prices[0] to prices[-1] 
            #     # 3. 
            #     roots = cs.roots(extrapolate=True)
            #     if len(roots) == 0 or roots[-1] < prices[-1]:
            #         slope = (beliefs[-1] - beliefs[0]) / (prices[-1] - prices[0])
            #         if slope == 0:
            #             #When only 1 element or all beliefs are same
            #             #in which case, cubic spline would also fail. 
            #             #do we need to consider such a case?
            #             input("rip")
            #         else:
            #             max_x = sp.optimize.minimize_scalar(optimize, bounds=(
            #                 float(prices[0]), prices[-1]), method='bounded')
            #             root = prices[-1] + (-beliefs[-1] / slope)
            #             if self.belief_function(root, SELL, last_L_orders) > 0:
            #                 print(root)
            #                 input(self.belief_function(root, SELL, last_L_orders))
            #             cs_end = sp.interpolate.interp1d([prices[-1], root], [beliefs[-1], 0])
            #             # fig,ax = plt.subplots()
            #             # cs_arr = []
            #             # val = prices[-1]
            #             # vals = []
            #             # increment = (root - prices[-1]) / 100
            #             # while val < root:
            #             #     cs_arr.append(float(cs_end(val)))
            #             #     vals.append(val)
            #             #     val += increment
            #             # plt.plot(vals, cs_arr)
            #             # ax.set_xlabel("Sell ask price")
            #             # ax.set_ylabel("Belief value")
            #             # plt.plot(vals, optimize_vals)
            #             # print(slope, root)
            #             # print("_________________")
            #             # print(prices, beliefs)
            #             # print(vals)
            #             # print("\n\n\n\n", cs_arr)
            #             #plt.show() 
            #             def optimize_end(price): return -((
            #                 price - self.pv.value_at_position(self.position) - self.estimate_fundamental()) * max(cs_end(price),0))
            #             max_x_end = sp.optimize.minimize_scalar(optimize_end, bounds=(
            #                 float(prices[-1]), root), method='bounded')
            #             #print("UPPER")
            #             if max_x.fun <= max_x_end.fun:
            #                 return max_x.x.item()
            #             else:
            #                 print(prices, beliefs)
            #                 input(max_x_end.x.item())
            #                 return max_x_end.x.item()
            #     else:
            #         root = roots[-1]
                
            # else:
            #     root = prices[-1]

            # max_x = sp.optimize.minimize_scalar(optimize, bounds=(
            #     float(prices[0]), root), method='bounded')
            
                # if max_x_end.x.item() > 1e10:
                #     print(max_x_end.x.item())  
                #     print(prices[-1], beliefs[-1])
                #     input()
                #     fig,ax = plt.subplots()
                #     cs_arr = []
                #     optimize_vals = []
                #     val = prices[-1]
                #     vals = []
                #     increment = 1000
                #     while val < prices[-1] + 1e6:
                #         cs_arr.append(float(cs_end(val)))
                #         optimize_vals.append(float(optimize_end(val)))
                #         vals.append(val)
                #         val += increment
                #     plt.plot(vals, cs_arr)
                #     ax.set_xlabel("Sell ask price")
                #     ax.set_ylabel("Belief value")
                #     # plt.plot(vals, optimize_vals)
                #     plt.show() 
            # print("LOWER", prices, root)
            # input(max_x.x.item())
            return max_x.x.item()            

    def take_action(self, side):
        '''
            Behavior reverts to ZI agent if L > total num of trades executed.
        '''
        # start_time = time.time()
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
            # print(time.time() - start_time)
            # if opt_price > 1e10:
            #     print(opt_price)
            #     print(self.market.order_book.sell_unmatched.heap)
            #     input("HELLo")
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

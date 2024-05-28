import random
import sys
import scipy as sp
import numpy as np
import time as timer
import matplotlib.pyplot as plt
from agent.agent import Agent
from market.market import Market
from fourheap.order import Order
from private_values.private_values import PrivateValues
from fourheap.constants import BUY, SELL
from typing import List
from fastcubicspline import FCS
from scipy.interpolate import CubicSpline


class HBLAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: List, L: int, pv_var: float,
                 arrival_rate: float, pv = None):
        self.agent_id = agent_id
        self.market = market
        if pv != None:
            self.pv = pv
        else:
            self.pv = PrivateValues(q_max, pv_var)
        self.position = 0
        self.shade = shade
        self.cash = 0
        self.L = L
        self.grace_period = 1 / arrival_rate
        self.lower_bound_mem = 0
        # self.obs_noise = obs_noise
        # self.prev_arrival_time = 0
        # self.prev_obs_mean = 0
        # self.prev_obs_var = 0
        
        # spoofing accuracy mid point
        self.mid_shade = 99/100
        self.half_shade = 1/2
        self.sell_half_shade = 1/2
        self.sell_mid_shade = 99/100
        self.prices_before_spoofer = []
        self.prices_after_spoofer = []
        self.sell_before_spoofer = []
        self.sell_after_spoofer = []
        self.sell_count = [0,0]
        self.buy_count = [0,0]

    def get_id(self) -> int:
        return self.agent_id

    # def noisy_obs(self):
    #     mean, r, T = self.market.get_info()
    #     t = self.market.get_time()
    #     val = self.market.get_fundamental_value()
    #     ot = val + np.random.normal(0,np.sqrt(self.obs_noise))

    #     rho_noisy = (1-r)**(t-self.prev_arrival_time)
    #     rho_var = rho_noisy ** 2

    #     prev_estimate = (1-rho_noisy)*mean + rho_noisy*self.prev_obs_mean
    #     prev_var =  rho_var * self.prev_obs_var + (1 - rho_var) / (1 - (1-r)**2) * int(self.market.fundamental.shock_std ** 2)

    #     curr_estimate = self.obs_noise / (self.obs_noise + prev_var) * prev_estimate + prev_var / (self.obs_noise + prev_var) * ot
    #     curr_var = self.obs_noise * prev_var / (self.obs_noise + prev_var)

    #     rho = (1-r)**(T-self.prev_arrival_time)

    #     self.prev_arrival_time = T
    #     self.prev_obs_mean = curr_estimate
    #     self.prev_obs_var = curr_var
        
    #     return (1 - rho) * mean + rho * curr_estimate

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()
        rho = (1 - r) ** (T - t)

        estimate = (1 - rho) * mean + rho * val
        # print(f'It is time {t} with final time {T} and I observed {val} and my estimate is {rho, estimate}')
        return estimate

    def find_worst_order(self, side, order_mem, orders: List[Order]):
        """
        Binary search to find the most competitive order in memory with a belief of 0.
        Args:
            side (int): Buy or Sell.
            order_mem (List[Order]): A sorted list of buy or sell orders.
                Both are sorted in ascending value of belief (for buy, ascending prices; for sell, descending prices)

        Returns:
            price: The price of the most competitive order with the a belief of 0.

            Note: we reverse order_mem for sells so that we can reuse code.
        """
        beginning = 0
        end = len(order_mem) - 1
        while beginning < end:
            mid = (beginning + end) // 2
            mid_belief = self.fast_belief_function(order_mem[mid].price, side, orders)
            if mid != len(order_mem) - 1:
                if mid_belief:
                    if not self.fast_belief_function(order_mem[mid + 1].price, side, orders):
                        return order_mem[mid].price, 0
                    if beginning == mid and mid_belief:
                        return order_mem[mid + 1].price, 0
                    beginning = mid
                else:
                    end = mid
            else:
                return mid
        return order_mem[0].price, self.belief_function(order_mem[0].price, side, orders)

    def get_last_trade_time_step(self):
        """
        Gets memory boundary time step based on L (how many matched orders considered in memory).

        Returns:
            timestep of earliest contributing order (i.e. the boundary timestep for memory).
        """
        # Assumes that matched_orders is ordered by timestep of trades
        last_matched_order_ind = len(self.market.matched_orders) - self.L * 2
        earliest_order = min(self.market.matched_orders[last_matched_order_ind:],
                             key=lambda matched_order: matched_order.order.time).order.time
        return earliest_order

    def fast_belief_function(self, p, side, orders):
        """
        To check if belief of order with price p is 0. Used for faster queries in find_worst_order()
        Args:
            p (float): price that will be checked.
            side (int): Buy or Sell.
            orders (List[Order]): Orders in memory

        Returns:
            bool: Whether or not belief is 0
        """
        start_time = timer.time()
        if side == BUY:
            TBL = 0  # Transact bids less or equal
            AL = 0  # Asks less or equal
            for ind, order in enumerate(orders):
                if order.price <= p and order.order_type == SELL:
                    AL += order.quantity
                for matched_order in self.market.matched_orders:
                    if order.order_id == matched_order.order.order_id:
                        if matched_order.order.order_type == BUY and matched_order.price <= p:
                            TBL += order.quantity
                        break
            return AL + TBL == 0
        else:
            TAG = 0  # Transact ask greater or equal
            BG = 0  # Bid greater or equal
            for ind, order in enumerate(orders):
                if order.price >= p and order.order_type == BUY:
                    BG += order.quantity
                for matched_order in self.market.matched_orders:
                    if order.order_id == matched_order.order.order_id:
                        if matched_order.order.order_type == SELL and matched_order.price >= p:
                            TAG += order.quantity
                        break
            return BG + TAG == 0

    def belief_function(self, p, side, orders):
        """
        Calculate belief of order with price p of transacting based on memory
        Args:
            p (float): price that will be checked.
            side (int): Buy or Sell.
            orders (List[Order]): Orders in memory

        Returns:
            float: Probability of order with price p transacting
        """
        current_time = self.market.get_time()
        if side == BUY:
            TBL = 0  # Transact bids less or equal
            AL = 0  # Asks less or equal
            RBG = 0  # Rejected bids greater or equal
            for ind, order in enumerate(orders):
                if order.price - p <= 0 and order.order_type == SELL:
                    AL += order.quantity
                found_matched = False
                for matched_order in self.market.matched_orders:
                    if order.order_id == matched_order.order.order_id:
                        if matched_order.order.order_type == BUY and matched_order.price - p <= 0:
                            TBL += order.quantity
                        found_matched = True
                        break
                if not found_matched:
                    if order.order_type == BUY and order.price - p >= 0:
                        # order time to withdrawal time
                        withdrawn = False
                        latest_order_time = 0
                        for i in range(ind + 1, len(orders)):
                            if orders[i].agent_id == order.agent_id and orders[i].order_id != order.order_id and orders[i].time > order.time:
                                latest_order_time = orders[i].time
                                withdrawn = True
                                break
                        if not withdrawn:
                            #Order not withdrawn
                            alive_time = current_time - order.time
                            if alive_time >= self.grace_period:
                                #Rejected
                                RBG += order.quantity
                            else:
                                #Partial rejection
                                RBG += (alive_time / self.grace_period) * order.quantity
                        else:
                            #Withdrawal
                            time_till_withdrawal = latest_order_time - order.time
                            #Withdrawal
                            if time_till_withdrawal >= self.grace_period:
                                RBG += order.quantity
                            else:
                                RBG += (time_till_withdrawal / self.grace_period) * order.quantity
                                
            if TBL + AL == 0:
                return 0
            else:
                return (TBL + AL) / (TBL + AL + RBG)

        else:
            TAG = 0  # Transact ask greater or equal
            BG = 0  # Bid greater or equal
            RAL = 0  # Reject ask less or equal

            for order in orders:
                if order.price - p >= 0 and order.order_type == BUY:
                    BG += order.quantity

            for ind, order in enumerate(orders):
                found_matched = False
                for matched_order in self.market.matched_orders:
                    if order.order_id == matched_order.order.order_id:
                        if matched_order.order.order_type == SELL and matched_order.price - p >= 0:
                            TAG += order.quantity
                        found_matched = True
                        break
                if not found_matched:
                    if order.order_type == SELL and order.price - p <= 0:
                        # order time to withdrawal time
                        withdrawn = False
                        latest_order_time = 0
                        for i in range(ind + 1, len(orders)):
                            if orders[i].agent_id == order.agent_id:
                                latest_order_time = orders[i].time
                                withdrawn = True
                                break
                        if not withdrawn:
                            alive_time = current_time - order.time
                            if alive_time >= self.grace_period:
                                RAL += order.quantity
                            else:
                                RAL += (alive_time / self.grace_period) * order.quantity
                        else:
                            time_till_withdrawal = latest_order_time - order.time
                            if time_till_withdrawal >= self.grace_period:
                                RAL += order.quantity
                            else:
                                RAL += (time_till_withdrawal / self.grace_period) * order.quantity
            if TAG + BG == 0:
                return 0
            else:
                return (TAG + BG) / (TAG + BG + RAL)
    
    def get_order_list(self):
        """
        Gets list of orders in memory. 
        
        Returns:
            last_L_orders: list of all orders in memory
            buy_orders_memory: filtered of last_L_orders with just BUY orders
            sell_orders_memory: filtered of last_L_orders with just SELL orders
        """
        self.lower_bound_mem = self.get_last_trade_time_step()

        buy_orders_memory = []
        sell_orders_memory = []
        last_L_orders = []
        for time in range(self.lower_bound_mem, self.market.get_time() + 1):
            last_L_orders.extend(self.market.event_queue.scheduled_activities[time])
        buy_orders_memory = [order for order in last_L_orders if order.order_type == BUY]
        sell_orders_memory = [order for order in last_L_orders if order.order_type == SELL]
        return last_L_orders, buy_orders_memory, sell_orders_memory

    # @profile
    def determine_optimal_price(self, side):
        """
        Determines optimal price for submission.
        Args:
            side (int): Buy or Sell.
        Returns:
            optimal price of submission and expected surplus weighted by probability of order transacting
        
        Useful references: https://www.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2009/notes/joel.pdf
            http://spider.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2005/notes/das.pdf 
        """

        last_L_orders, buy_orders_memory, sell_orders_memory = self.get_order_list()
        last_L_orders = np.array(last_L_orders)
        estimate = self.estimate_fundamental()
        buy_orders_memory = sorted(buy_orders_memory, key = lambda order:order.price)
        sell_orders_memory = sorted(sell_orders_memory, key = lambda order:order.price)
        best_ask = float(self.market.order_book.sell_unmatched.peek())
        best_buy = float(self.market.order_book.buy_unmatched.peek())
        #First is interpolate objects. Second is corresponding bounds
        spline_interp_objects = [[], []]
        if side == BUY: 
            private_value = self.pv.value_for_exchange(self.position, BUY)
            best_buy_belief = self.belief_function(best_buy, BUY, last_L_orders)
            best_ask_belief = 1
            def interpolate(bound1, bound2, bound1Belief, bound2Belief):
                cs = FCS(bound1, bound2, [bound1Belief, bound2Belief])
                # cs = CubicSpline([bound1, bound2], [bound1Belief, bound2Belief])
                spline_interp_objects[0].append(cs)
                spline_interp_objects[1].append((bound1, bound2))

            # @profile
            def expected_surplus_max():
                """
                Calculates price with maximum expected surplus.

                Returns:
                    Optimal price and corresponding expected surplus.
                """
                # @profile
                def optimize(price): 
                    """
                    Calculates price with maximum expected surplus.
                    
                    Params:
                        price: Price 

                    Returns:
                        Returns expected surplus of price p.
                    """
                    for i in range(len(spline_interp_objects[0])):
                        # Spline interpolation objects is an array of interpolations over the entire domain. 
                        # There's a different interpolation function for each continuous partition of the domain. 
                        # (I.e. function is piecewise continuous)
                        if spline_interp_objects[1][i][0] <= price <= spline_interp_objects[1][i][1]:
                            return -((estimate + private_value - price) * spline_interp_objects[0][i](price))

                lb = min(spline_interp_objects[1], key=lambda bound_pair: bound_pair[0])[0]
                ub = max(spline_interp_objects[1], key=lambda bound_pair: bound_pair[1])[1]
                # print(lb,ub)
                # x = np.linspace(lb, ub, 500)
                # vals = [-optimize(val) for val in x]
                # plt.plot(x, vals)
                # plt.xlabel('x')
                # plt.ylabel('Optimized Values')
                # plt.title('Optimization Results')
                # plt.grid(True)
                # plt.show()

                # Because function (when graphed) is well defined to be unimodal, we select 
                # many test points and then local optimize based on best point. 
                # Saves time as opposed to global optimizing.
                test_points = np.linspace(lb, ub, 40)
                vOptimize = np.vectorize(optimize)
                point_surpluses = vOptimize(test_points)
                min_index = np.argmin(point_surpluses)
                min_survey = test_points[min_index]
                # max_x = min_survey, -np.min(point_surpluses)
                
                max_x = sp.optimize.minimize(vOptimize, min_survey, bounds=[[lb, ub]])
                
                
                # if self.market.get_time() > 5000:
                # plt.plot(test_points, point_surpluses, marker='o', color="red", linestyle='-')
                # plt.scatter(max_x.x.item(), max_x.fun, color="black", s=15, zorder=3)
                # plt.scatter(estimate, 0, color="blue", s=15, zorder=3)
                # plt.scatter(estimate + private_value, 0, color="green", s=15, zorder=2)
                # plt.xlabel('Test Points')
                # plt.ylabel('Points')
                # plt.title('Surplus v Test Points - BUY')
                # plt.grid(True)
                # plt.show()
                # return max_x
                return max_x.x.item(), -max_x.fun

            buy_high = float(buy_orders_memory[-1].price)
            buy_high_belief = self.belief_function(buy_high, BUY, last_L_orders)
            buy_low, buy_low_belief = self.find_worst_order(BUY, buy_orders_memory, last_L_orders)
            optimal_price = (0,-sys.maxsize)

            if buy_high >= best_ask:
                buy_high = best_ask
                buy_high_belief = best_ask_belief
                buy_low = min(buy_high, buy_low)
                buy_low_belief = min(buy_high_belief, buy_low_belief)
            
            #Best ask > buy high >= best_buy
            if buy_high >= best_buy:
                #interpolate between best ask and buy high
                if best_ask != buy_high:
                    interpolate(buy_high, best_ask, buy_high_belief, 1)
                if best_buy >= buy_low:
                    buy_mid = buy_low + self.mid_shade * abs(best_buy - buy_low)
                    buy_mid_belief = self.belief_function(buy_mid, BUY, last_L_orders)
                    buy_half = buy_low + self.half_shade * abs(best_buy - buy_low)
                    buy_half_belief = self.belief_function(buy_half, BUY, last_L_orders)
                    if best_buy != buy_high:
                        #interpolate between best buy and buy_high 
                        interpolate(best_buy, buy_high, best_buy_belief, buy_high_belief)
                    if best_buy != buy_mid:
                        #if best_buy != buy_mid, best_buy > buy_low
                        #interpolate between best buy and buy_mid (for accuracy on spoofing)
                        interpolate(buy_low, buy_half, buy_low_belief, buy_half_belief)
                        interpolate(buy_half, buy_mid, buy_half_belief, buy_mid_belief)
                        interpolate(buy_mid, best_buy, buy_mid_belief, best_buy_belief)
                    if buy_low_belief > 0:
                        #interpolate between buy_low and 0
                        lower_bound = max(buy_low - 2 * (buy_high - buy_low) - 1, 0)
                        interpolate(lower_bound, buy_low, 0, buy_low_belief)
                elif best_buy < buy_low:
                    #interpolate between buy_high and buy_low
                    if buy_high != buy_low:
                        interpolate(buy_low, buy_high, buy_low_belief, buy_high_belief)
                    #interpolate buy_low and best_buy
                    if buy_low != best_buy:
                        interpolate(best_buy, buy_low, best_buy_belief, buy_low_belief)
                    #interpolate best_buy and 0?
                    if best_buy_belief > 0:
                        lower_bound = max(best_buy - 2 * (buy_high - best_buy) - 1,0)
                        buy_mid = lower_bound + self.mid_shade * abs(best_buy - lower_bound)
                        buy_mid_belief = self.belief_function(buy_mid, BUY, last_L_orders)
                        buy_half = lower_bound + self.half_shade * abs(best_buy - lower_bound)
                        buy_half_belief = self.belief_function(buy_half, BUY, last_L_orders)
                        interpolate(buy_mid, best_buy, buy_mid_belief, best_buy_belief)
                        interpolate(buy_half, buy_mid, buy_half_belief, buy_mid_belief)
                        interpolate(lower_bound, buy_half, 0, buy_half_belief)
                        

            elif buy_high < best_buy:
                buy_mid = buy_high + self.mid_shade * abs(best_buy - buy_high)
                buy_mid_belief = self.belief_function(buy_mid, BUY, last_L_orders)
                buy_half = buy_high + self.half_shade * abs(best_buy - buy_high)
                buy_half_belief = self.belief_function(buy_half, BUY, last_L_orders)
                # interpolate between best_ask and best_buy
                # occasionally have bug where best buy is > best_ask? see slurm-6885793
                if best_ask != best_buy:
                    interpolate(best_buy, best_ask, best_buy_belief, best_ask_belief)
                #interpolate between best_buy and buy_high
                if best_buy != buy_high:
                    interpolate(buy_high, buy_half, buy_high_belief, buy_half_belief)
                    interpolate(buy_half, buy_mid, buy_half_belief, buy_mid_belief)
                    interpolate(buy_mid, best_buy, buy_mid_belief, best_buy_belief)
                    
                #interpolate between buy_high and buy_low
                if buy_high != buy_low:
                    interpolate(buy_low, buy_high, buy_low_belief, buy_high_belief)
                    
                #interpolate buy_low and 0
                if buy_low_belief > 0:
                    #TODO: Might reconsider this bound. If buy high is quite high, this bound distance
                    # could be very far. Could be an issue
                    lower_bound = max(buy_low - 2 * (buy_high - buy_low) - 1, 0)
                    interpolate(lower_bound, buy_low, 0, buy_low_belief)

            optimal_price = expected_surplus_max()
            
            if optimal_price == (0,0):
                interpolate(buy_low, buy_high, 0, 1)
                optimal_price = expected_surplus_max()
                print("BUY", buy_low, buy_low_belief, buy_high, buy_high_belief, best_buy, best_buy_belief, best_ask)
                print(optimal_price)
                input("ERROR")
            # Edge case: If a lot of orders have expected surplus of 0 (meaning belief of 0),
            # at least submit order that doesn't lose agent money in the edge case
            # that the order submits even if it has belief of 0. 
            if optimal_price[0] > estimate + private_value:
                return estimate + private_value, -1
            
            return optimal_price[0], optimal_price[1]

        else:
            private_value = self.pv.value_for_exchange(self.position, SELL)
            best_buy_belief = 1
            best_ask_belief = self.belief_function(best_ask, SELL, last_L_orders)
            sell_high, sell_high_belief = self.find_worst_order(SELL, sorted(sell_orders_memory, key=lambda order: order.price, reverse=True), last_L_orders)
            optimal_price = (0,-sys.maxsize)
            best_buy_belief = 1
            sell_low = float(sell_orders_memory[0].price)
            sell_low_belief = self.belief_function(sell_low, SELL, last_L_orders)
            def interpolate(bound1, bound2, bound1Belief, bound2Belief):
                """
                Sell version of interpolate above. 
                @TODO: Merge the two
                """
                cs = FCS(bound1, bound2, [bound1Belief, bound2Belief])
                # cs = CubicSpline([bound1, bound2], [bound1Belief, bound2Belief])
                spline_interp_objects[0].append(cs)
                if bound2 > 1e10:
                    print(bound2)
                    print(bound1)
                    input("ERROR")
                spline_interp_objects[1].append((bound1, bound2))
                
            def expected_surplus_max():
                """
                Sell version of the same function above in BUY. 
                @TODO: Merge the two
                """
                def optimize(price): 
                    """
                    Sell version of the same function above in BUY. 
                    @TODO: Merge the two
                    """
                    for i in range(len(spline_interp_objects[0])):
                        if spline_interp_objects[1][i][0] <= price <= spline_interp_objects[1][i][1]:
                            return -((price - (estimate + private_value)) * spline_interp_objects[0][i](price))

                lb = min(spline_interp_objects[1], key=lambda bound_pair: bound_pair[0])[0]
                ub = max(spline_interp_objects[1], key=lambda bound_pair: bound_pair[1])[1]
                test_points = np.linspace(lb, ub, 40)
                # try:
                vOptimize = np.vectorize(optimize)
                point_surpluses = vOptimize(test_points)
                min_index = np.argmin(point_surpluses)
                min_survey = test_points[min_index]
                # max_x = min_survey, -np.min(point_surpluses)
                #REINSTATE AFTER
                max_x = sp.optimize.minimize(vOptimize, min_survey, bounds=[[lb, ub]])
                # plt.plot(test_points, point_surpluses, marker='o', linestyle='-',color="cyan")
                # plt.scatter(max_x.x.item(), max_x.fun, color="black", s= 15, zorder=3)
                # plt.scatter(estimate, 0, color="blue", s=15, zorder=3)
                # plt.scatter(estimate + private_value, 0, color="green", s=15, zorder=2)
                # plt.xlabel('Test Points')
                # plt.ylabel('Points')
                # plt.title('Surplus v Test Points - SELL')
                # plt.grid(True)
                # plt.show()
                # return max_x
                return max_x.x.item(), -max_x.fun

            if best_buy > sell_low:
                sell_low = best_buy
                sell_low_belief = 1
                sell_high = max(sell_high, sell_low)
                sell_high_belief = min(sell_high_belief, sell_low_belief)

            if sell_low <= best_ask:
                # interpolate best buy to sell_low
                if sell_low != best_buy:
                    interpolate(best_buy, sell_low, best_buy_belief, sell_low_belief)
                if best_ask <= sell_high:
                    if sell_low != best_ask:
                        sell_mid = sell_low + self.sell_mid_shade * abs(best_ask - sell_low)
                        sell_mid_belief = self.belief_function(sell_mid, SELL, last_L_orders)
                        sell_half = sell_low + self.sell_half_shade * abs(best_ask - sell_low)
                        sell_half_belief = self.belief_function(sell_half, SELL, last_L_orders)
                        #interpolate sell_low to sell_mid
                        if sell_low != sell_half:
                            interpolate(sell_low, sell_half, sell_low_belief, sell_half_belief)
                        #interpolate sell_half to sell_mid
                        if sell_half != sell_mid:
                            interpolate(sell_half, sell_mid, sell_half_belief, sell_mid_belief)
                        #interpolate sell_mid to best_ask
                        if sell_mid != best_ask:
                            interpolate(sell_mid, best_ask, sell_mid_belief, best_ask_belief)
                    if best_ask != sell_high:
                        #interpolate best_ask to sell_high
                        interpolate(best_ask, sell_high, best_ask_belief, sell_high_belief)
                        
                    # interpolate sell_high to ????
                    if sell_high_belief > 0:
                        upper_bound = sell_high + 2 * (sell_high - best_buy) + 1
                        interpolate(sell_high, upper_bound, sell_high_belief, 0)
                        
                elif best_ask > sell_high:
                    if sell_low != sell_high:
                        #interpolate low sell to high sell
                        interpolate(sell_low, sell_high, sell_low_belief, sell_high_belief)

                    if sell_high != best_ask:
                        sell_mid = sell_high + self.sell_mid_shade * abs(best_ask - sell_high)
                        sell_mid_belief = self.belief_function(sell_mid, SELL, last_L_orders)
                        sell_half = sell_high + self.sell_half_shade * abs(best_ask - sell_high)
                        sell_half_belief = self.belief_function(sell_half, SELL, last_L_orders)
                        #interpolate sell_high to sell_mid
                        if sell_high != sell_half:
                            interpolate(sell_high, sell_half, sell_high_belief, sell_half_belief)
                        if sell_half != sell_mid:
                            interpolate(sell_half, sell_mid, sell_half_belief, sell_mid_belief)
                        #interpolate sell_high to best ask
                        if sell_mid != best_ask:
                            interpolate(sell_mid, best_ask, sell_mid_belief, best_ask_belief)
                        
                    #interpolate sell_high to sell_high + 2*spread
                    if best_ask_belief > 0:
                        upper_bound = best_ask + 2 * (best_ask - best_buy) + 1
                        interpolate(best_ask, upper_bound, best_ask_belief, 0)
                        
            elif sell_low > best_ask:
                if best_buy != best_ask:
                    sell_mid = best_buy + self.sell_mid_shade * abs(best_ask - best_buy)
                    sell_mid_belief = self.belief_function(sell_mid, SELL, last_L_orders)
                    sell_half = best_buy + self.sell_half_shade * abs(best_ask - best_buy)
                    sell_half_belief = self.belief_function(sell_half, SELL, last_L_orders)
                    #interpolate best_buy to best_ask
                    interpolate(best_buy, sell_half, best_buy_belief, sell_half_belief)
                    interpolate(sell_half, sell_mid, sell_half_belief, sell_mid_belief)
                    #interpolate sell_mid to best_ask
                    interpolate(sell_mid, best_ask, sell_low_belief, best_ask_belief)
                if best_ask != sell_low:
                    #interpolate best_ask to sell_low
                    interpolate(best_ask, sell_low, best_ask_belief, sell_low_belief)
                    
                if sell_low != sell_high:
                    #interpolate sell_low to sell_high
                    interpolate(sell_low, sell_high, sell_low_belief, sell_high_belief)
                    
                #interpolate sell_high to sell_high + 2*spread
                if sell_high_belief > 0:
                    upper_bound = sell_high + 2 * (sell_high - best_buy) + 1
                    interpolate(sell_high, upper_bound, sell_high_belief, 0)
            
            optimal_price = expected_surplus_max()

            if optimal_price == (0,0):
                print("SELL", sell_low, sell_low_belief, sell_high, sell_high_belief, best_ask, best_buy)
                interpolate(sell_low, sell_high, 1, 0)
                optimal_price = expected_surplus_max()
                print(optimal_price)
                input("ERROR")
            
            # x = self.belief_function(best_ask + 1,SELL, last_L_orders)
            # z = self.belief_function(min(sell_low, best_ask) + 3/4 * abs(sell_low - best_ask), SELL, last_L_orders)
            # for i in range(len(spline_interp_objects[0])):
            #     if spline_interp_objects[1][i][0] <= best_ask + 1 <= spline_interp_objects[1][i][1]:
            #         a = spline_interp_objects[0][i](best_ask + 1)
            #     if spline_interp_objects[1][i][0] <= min(sell_low, best_ask) + 3/4 * abs(sell_low - best_ask)<= spline_interp_objects[1][i][1]:
            #         b = spline_interp_objects[0][i](min(sell_low, best_ask) + 3/4 * abs(sell_low - best_ask))
               
            
            #EDGE CASE (SAME AS ABOVE IN BUY)
            if optimal_price[0] < estimate + private_value:
                return estimate + private_value, 0
            
           
            a = self.belief_function(optimal_price[0], SELL, last_L_orders)
            return optimal_price[0], optimal_price[1]

    def take_action(self, side, seed = None):
        """
        Submits order to market for HBL.

        Params:
            side: BUY or SELL.

        Returns:
            order [Order]: order to be submitted

        Note:
            Behavior reverts to ZI agent if L > total num of trades executed.
        """
        t = self.market.get_time()
        random.seed(t + seed)
        estimate = self.estimate_fundamental()
        spread = self.shade[1] - self.shade[0]
        if len(self.market.matched_orders) >= 2 * self.L and self.market.order_book.buy_unmatched.peek_order() != None and self.market.order_book.sell_unmatched.peek_order() != None:
            opt_price, opt_price_est_surplus = self.determine_optimal_price(side)
            if t > 1000:
                print(f'It is time {t} and I am on {side} as an HBL. My estimate is {estimate}, my position is {self.position}, and my marginal pv is '
                f'{self.pv.value_for_exchange(self.position, side)}.'
                f'Therefore I offer price {opt_price}')
    
            order = Order(
                price=opt_price,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=side,
                order_id=random.randint(1, 10000000)
            )
            # print("HBL")
            # input(order)
            
            return [order]

        else:
            # ZI Agent
            valuation_offset = spread*random.random() + self.shade[0]
            if side == BUY:
                price = estimate + self.pv.value_for_exchange(self.position, BUY) - valuation_offset
            elif side == SELL:
                price = estimate + self.pv.value_for_exchange(self.position, SELL) + valuation_offset
            # if t < 1050:
            #     print(f'It is time {t} and I am HBL but act as ZI on {side}. My estimate is {estimate} and my marginal pv is '
            #         f'{self.pv.value_for_exchange(self.position, side)} with offset {valuation_offset}. '
            #         f'Therefore I offer price {price}')
            order = Order(
                price=price,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=side,
                order_id=random.randint(1, 10000000)
            )
            return [order]

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def __str__(self):
        return f'HBL{self.agent_id}'

    def reset(self):
        self.position = 0
        self.cash = 0

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)
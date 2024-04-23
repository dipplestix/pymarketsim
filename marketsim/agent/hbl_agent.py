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

class HBLAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: List, L: int, pv_var: float, arrival_rate : float):
        self.agent_id = agent_id
        self.market = market
        self.pv = PrivateValues(q_max, pv_var)
        self.position = 0
        self.shade = shade
        self.cash = 0
        self.L = L
        self.grace_period = 1/arrival_rate
        self.order_history = None
        self.lower_bound_mem = 0

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*val
        # print(f'It is time {t} with final time {T} and I observed {val} and my estimate is {rho, estimate}')
        return estimate

    def get_last_trade_time_step(self):
        '''
            Gets time step boundary for orders occurring since the earliest 
            order contributing to the Lth most recent trade up to most recent trade time.
        '''
        # Assumes that matched_orders is ordered by timestep of trades
        last_matched_order_ind = len(self.market.matched_orders) - self.L*2
        #most_recent_trade = max(
        #    self.market.matched_orders[-1].order.time, self.market.matched_orders[-2].order.time)
        # Gets earliest contributing buy/sell
        earliest_order = min(self.market.matched_orders[last_matched_order_ind:], key=lambda matched_order: matched_order.order.time).order.time
        return earliest_order

    def fast_belief_function(self, p, side, orders):
        if side == BUY:
            normalized_orders = np.array([(order.price, order.order_id, order.order_type, order.agent_id, order.time) for order in orders])
            buy_matched_orders = np.array([(matched_order.price, matched_order.order.order_id) for matched_order in self.market.matched_orders 
                                                if matched_order.order.order_type == BUY and matched_order.order.time >= self.lower_bound_mem])
            sell_orders = normalized_orders[normalized_orders[:, 2] == SELL]
            AL = np.sum(sell_orders[:, 0] <= p)
            TBL = np.sum(buy_matched_orders <= p)
            return AL + TBL == 0
        else:
            normalized_orders = np.array([(order.price, order.order_id, order.order_type, order.agent_id, order.time) for order in orders])
            sell_matched_orders = np.array([(matched_order.price, matched_order.order.order_id) for matched_order in self.market.matched_orders 
                                                if matched_order.order.order_type == SELL and matched_order.order.time >= self.lower_bound_mem])
            buy_orders = normalized_orders[normalized_orders[:, 2] == BUY]
            BG = np.sum(buy_orders[:, 0] >= p)
            TAG = np.sum(sell_matched_orders >= p)
            return BG + TAG == 0

    def belief_function(self, p, side, orders):
        
        '''
            Defined over all past bid/ask prices in orders. 
            Need cubic spline interpolation for prices not encountered previously.
            Returns: probability that bid/ask will execute at price p.
            TODO: Optimize? And optimize for history changes as orders are added
        '''
        start_time = timer.time()
        current_time = self.market.get_time()
        if side == BUY:
            start_time = timer.time()
            # TBL: Transact bids less or equal
            # AL: Asks less or equal
            # RBG: Rejected bids greater or equal
            normalized_orders = np.array([(order.price, order.order_id, order.order_type, order.agent_id, order.time) for order in orders])
            buy_matched_orders = np.array([(matched_order.price, matched_order.order.order_id) for matched_order in self.market.matched_orders 
                                                if matched_order.order.order_type == BUY and matched_order.order.time >= self.lower_bound_mem])
            sell_orders = normalized_orders[normalized_orders[:, 2] == SELL]
            AL = np.sum(sell_orders[:, 0] <= p)
            TBL = np.sum(buy_matched_orders[:, 0] <= p)
            buy_orders = normalized_orders[normalized_orders[:, 2] == BUY]
            buy_orders_greater_p = buy_orders[buy_orders[:, 0] >= p]
            unmatched_bids_greater_p = buy_orders_greater_p[np.in1d(buy_orders_greater_p[:, 1], buy_matched_orders[:, 1], invert=True)]
            time_diffs = current_time - unmatched_bids_greater_p[:, 4]
            # Calculate grace period violations
            grace_period_violations = (time_diffs >= self.grace_period)

            # Calculate RBG contributions
            RBG_contributions = np.where(grace_period_violations, 1, time_diffs / self.grace_period)
            for i in range(len(unmatched_bids_greater_p) - 1):
                #Given agent submitted another order
                found_new_order = np.argwhere((unmatched_bids_greater_p[i, 3] == normalized_orders[:, 3]) & 
                                                (unmatched_bids_greater_p[i,1] != normalized_orders[:,1]) &
                                                (unmatched_bids_greater_p[i,4] < normalized_orders[:, 4]))
                if len(found_new_order) > 0: 
                    withdrawal_time = (normalized_orders[found_new_order[0], 4] - unmatched_bids_greater_p[i, 4])
                    RBG_contributions[i] = np.where(withdrawal_time >= self.grace_period, 1, withdrawal_time / self.grace_period)

            RBG = np.sum(RBG_contributions)
            print("BUY BELIEF TIME", timer.time() - start_time)
            print(TBL, AL, RBG)
            input()
            if TBL + AL == 0:
                return 0
            else:
                return (TBL + AL) / (TBL + AL + RBG)
        else:
            # TAG: Transact ask greater or equal
            # BG: Bid greater or equal
            # RAL: Reject ask less or equal
            normalized_orders = np.array([(order.price, order.order_id, order.order_type, order.agent_id, order.time) for order in orders])
            sell_matched_orders = np.array([(matched_order.price, matched_order.order.order_id) for matched_order in self.market.matched_orders 
                                                if matched_order.order.order_type == SELL and matched_order.order.time >= self.lower_bound_mem])
            buy_orders = normalized_orders[normalized_orders[:, 2] == BUY]
            BG = np.sum(buy_orders[:, 0] >= p)
            TAG = np.sum(sell_matched_orders >= p)
            sell_orders = normalized_orders[normalized_orders[:, 2] == SELL]
            sell_orders_less_p = sell_orders[sell_orders[:, 0] <= p]
            unmatched_asks_less_p = sell_orders_less_p[np.in1d(sell_orders_less_p[:, 1], sell_matched_orders[:, 1], invert=True)]
            time_diffs = current_time - unmatched_asks_less_p[:, 4]
            # Calculate grace period violations
            grace_period_violations = (time_diffs >= self.grace_period)

            # Calculate RBG contributions
            RAL_contributions = np.where(grace_period_violations, 1, time_diffs / self.grace_period)
            for i in range(len(unmatched_asks_less_p) - 1):
                #Given agent submitted another order
                found_new_order = np.argwhere((unmatched_asks_less_p[i, 3] == normalized_orders[:, 3]) & 
                                                (unmatched_asks_less_p[i,1] != normalized_orders[:,1]) &
                                                (unmatched_asks_less_p[i,4] < normalized_orders[:, 4]))
                if len(found_new_order) > 0: 
                    withdrawal_time = (normalized_orders[found_new_order[0], 4] - unmatched_asks_less_p[i, 4])
                    RAL_contributions[i] = np.where(withdrawal_time >= self.grace_period, 1, withdrawal_time / self.grace_period)

            RAL = np.sum(RAL_contributions)
            if TAG + BG == 0:
                return 0
            else:
                return (TAG + BG) / (TAG + BG + RAL)
    
    def get_order_list(self):
        self.lower_bound_mem = self.get_last_trade_time_step()

        buy_orders_memory = []
        sell_orders_memory = []
        last_L_orders = []
        for time in range(self.lower_bound_mem, self.market.get_time() + 1):
            for order in self.market.event_queue.scheduled_activities[time]:
                last_L_orders.append(order)
                if order.order_type == BUY:
                    buy_orders_memory.append(order)
                else:
                    sell_orders_memory.append(order)
        return last_L_orders, buy_orders_memory, sell_orders_memory

    @profile
    def determine_optimal_price(self, side):
        '''
            Reference: https://www.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2009/notes/joel.pdf
            http://spider.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2005/notes/das.pdf 

            Spline time is ~__% of the run time. ___/___
        '''
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
                start_time = timer.time()
                cs = FCS(bound1, bound2, [bound1Belief, bound2Belief])
                spline_interp_objects[0].append(cs)
                spline_interp_objects[1].append((bound1, bound2))
                # print("INTERPOLATE TIME", timer.time() - start_time)
            @profile
            def expected_surplus_max():
                @profile
                def optimize(price): 
                    start_time = timer.time()
                    for i in range(len(spline_interp_objects[0])):
                        #spline interp objects is an array of interpolations over the entire domain. 
                        # There's a different interpolation function for each continuous partitions of the domain.
                        if spline_interp_objects[1][i][0] <= price <= spline_interp_objects[1][i][1]:
                            x = -((estimate + private_value - price) * spline_interp_objects[0][i](price))
                            # print("TIMER", timer.time() - start_time)
                            return x
                start_time = timer.time()
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
                test_points = np.linspace(lb, ub, 20)
                vOptimize = np.vectorize(optimize)
                min_survey = np.min(vOptimize(test_points))
                max_x = sp.optimize.minimize(vOptimize, min_survey, bounds=[[lb, ub]])
                # input(max_x)
                # print("BUY DE MAX TIME", timer.time() - start_time)
                # input()
                return max_x.x.item(), -max_x.fun

            start_time = timer.time()
            buy_high = float(buy_orders_memory[-1].price)
            buy_high_belief = self.belief_function(buy_high, BUY, last_L_orders)
            i = 0
            for ind, order in enumerate(buy_orders_memory):
                if not self.belief_function(order.price, BUY, last_L_orders) and ind > 0:
                    i = ind - 1
                    break

            buy_low = buy_orders_memory[i - 1].price if i != 0 else buy_orders_memory[i].price
            buy_low_belief = self.belief_function(buy_orders_memory[i].price, BUY, last_L_orders)

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
                    if best_buy != buy_high:
                        #interpolate between best buy and buy_high 
                        interpolate(best_buy, buy_high, best_buy_belief, buy_high_belief)
                    if best_buy != buy_low:
                        #interpolate between best buy and buy_low
                        interpolate(buy_low, best_buy, buy_low_belief, best_buy_belief)
                    #interpolate between buy_low and 0
                    if buy_low_belief > 0:
                        lower_bound = buy_low - 2 * (best_ask - buy_low)
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
                        lower_bound = best_buy - 2 * (best_ask - best_buy)
                        interpolate(lower_bound, best_buy, 0, best_buy_belief)

            elif buy_high < best_buy:
                # interpolate between best_ask and best_buy
                # occasionally have bug where best buy is > best_ask? see slurm-6885793
                if best_ask != best_buy:
                    interpolate(best_buy, best_ask, best_buy_belief, best_ask_belief)
                #interpolate between best_buy and buy_high
                if best_buy != buy_high:
                    interpolate(buy_high, best_buy, buy_high_belief, best_buy_belief)
                    
                #interpolate between buy_high and buy_low
                if buy_high != buy_low:
                    interpolate(buy_low, buy_high, buy_low_belief, buy_high_belief)
                    
                #interpolate buy_low and 0
                if buy_low_belief > 0:
                    lower_bound = buy_low - 2 * (best_ask - buy_low)
                    interpolate(lower_bound, buy_low, 0, buy_low_belief)

            # print("BUY REST TIME", timer.time() - start_time)
            optimal_price = expected_surplus_max()
            # input(optimal_price)
            if optimal_price == (0,0):
                print("BUY", buy_low, buy_low_belief, buy_high, buy_high_belief, best_buy, best_buy_belief, best_ask)
                input("ERROR")
            #Adjusting for multiple 0 values (from the function). Edge case in case order with belief = 0 transacts.
            if optimal_price[0] > estimate + private_value:
                return estimate + private_value, 0
            
            return optimal_price[0], optimal_price[1]

        else:
            start_time = timer.time()
            private_value = self.pv.value_for_exchange(self.position, SELL)
            best_buy_belief = 1
            best_ask_belief = self.belief_function(best_ask, SELL, last_L_orders)
            i = len(sell_orders_memory) - 1
            for ind, order in reversed(list(enumerate(sell_orders_memory))):
                if not self.fast_belief_function(order.price, BUY, last_L_orders) and ind != len(sell_orders_memory) - 1:
                    i = ind + 1
                    break

            sell_high = sell_orders_memory[i].price
            sell_high_belief = self.belief_function(sell_orders_memory[i].price, SELL, last_L_orders)
            
            optimal_price = (0,-sys.maxsize)
            best_buy_belief = 1
            sell_low = float(sell_orders_memory[0].price)
            sell_low_belief = self.belief_function(sell_low, SELL, last_L_orders)
            def interpolate(bound1, bound2, bound1Belief, bound2Belief):
                #start_time = timer.time()
                #cs = sp.interpolate.CubicSpline([bound1, bound2], [bound1Belief, bound2Belief], extrapolate=False)
                cs = FCS(bound1, bound2, [bound1Belief, bound2Belief])
                spline_interp_objects[0].append(cs)
                spline_interp_objects[1].append((bound1, bound2))
                #print("INTERPOLATE TIME", timer.time() - start_time)
            
            def expected_surplus_max():
                def optimize(price): 
                    for i in range(len(spline_interp_objects[0])):
                        if spline_interp_objects[1][i][0] <= price <= spline_interp_objects[1][i][1]:
                            return -((price - (estimate + private_value)) * spline_interp_objects[0][i](price))
                    input("ERROR")
                start_time = timer.time()
                # print("MAX TIME", timer.time() - start_time)
                # input(stats)
                start_time = timer.time()
                #max_x = sp.optimize.differential_evolution(optimize, [[bound1, bound2]], maxiter=5)
                lb = min(spline_interp_objects[1], key=lambda bound_pair: bound_pair[0])[0]
                ub = max(spline_interp_objects[1], key=lambda bound_pair: bound_pair[1])[1]
                # print(lb,ub)
                # input(spline_interp_objects)
                test_points = np.linspace(lb, ub, 30)
                vOptimize = np.vectorize(optimize)
                min_survey = np.min(vOptimize(test_points))
                max_x = sp.optimize.minimize(vOptimize, min_survey, bounds=[[lb, ub]])
                # print("DE MAX TIME", timer.time() - start_time)
                # start_time = timer.time()
                # max_x_check = sp.optimize.differential_evolution(optimize, [[bound1, bound2]])
                # input() 
                return max_x.x.item(), -max_x.fun

            if best_buy > sell_low:
                sell_low = best_buy
                sell_low_belief = 1
                sell_high = max(sell_high, sell_low)
                sell_high_belief = min(sell_high_belief, sell_low_belief)

            if sell_low <= best_ask:
                #interpolate best buy to sell_low
                if sell_low != best_buy:
                    interpolate(best_buy, sell_low, best_buy_belief, sell_low_belief)
                if best_ask <= sell_high:
                    if sell_low != best_ask:
                        #interpolate sell_low to best_ask
                        interpolate(sell_low, best_ask, sell_low_belief, best_ask_belief)
                    if best_ask != sell_high:
                        #interpolate best_ask to sell_high
                        interpolate(best_ask, sell_high, best_ask_belief, sell_high_belief)
                        
                    # interpolate sell_high to ????
                    if sell_high_belief > 0:
                        upper_bound = sell_high + 2 * (sell_high - best_buy)
                        interpolate(sell_high, upper_bound, sell_high_belief, 0)
                        
                elif best_ask > sell_high:
                    if sell_low != sell_high:
                        #interpolate low sell to high sell
                        interpolate(sell_low, sell_high, sell_low_belief, sell_high_belief)

                    if sell_high != best_ask:
                        #interpolate sell_high to best ask
                        interpolate(sell_high, best_ask, sell_high_belief, best_ask_belief)
                        
                    #interpolate sell_high to sell_high + 2*spread
                    if best_ask_belief > 0:
                        upper_bound = best_ask + 2 * (best_ask - best_buy)
                        interpolate(best_ask, upper_bound, best_ask_belief, 0)
                        
            elif sell_low > best_ask:
                if best_buy != best_ask:
                    #interpolate best_buy to best_ask
                    interpolate(best_buy, best_ask, best_buy_belief, best_ask_belief)
                if best_ask != sell_low:
                    #interpolate best_ask to sell_low
                    interpolate(best_ask, sell_low, best_ask_belief, sell_low_belief)
                    
                if sell_low != sell_high:
                    #interpolate sell_low to sell_high
                    interpolate(sell_low, sell_high, sell_low_belief, sell_high_belief)
                    
                #interpolate sell_high to sell_high + 2*spread
                if sell_high_belief > 0:
                    upper_bound = sell_high + 2 * (sell_high - best_buy)
                    interpolate(sell_high, upper_bound, sell_high_belief, 0)
            
            # print("SELL REST TIME", timer.time() - start_time)
            # input(spline_interp_objects)
            optimal_price = expected_surplus_max()

            if optimal_price == (0,0):
                print("SELL", sell_low, sell_low_belief, sell_high, sell_high_belief, best_ask, best_buy)
                input("ERROR")
            #EDGE CASE
            if optimal_price[0] < estimate + private_value:
                return estimate + private_value, 0
            return optimal_price[0], optimal_price[1]

    def take_action(self, side):
        '''
            Behavior reverts to ZI agent if L > total num of trades executed.
        '''
        t = self.market.get_time()
        estimate = self.estimate_fundamental()
        spread = self.shade[1] - self.shade[0]
        start_time = timer.time()
        if len(self.market.matched_orders) >= 2 * self.L and not self.market.order_book.buy_unmatched.is_empty() and not self.market.order_book.sell_unmatched.is_empty():
            opt_price, opt_price_est_surplus = self.determine_optimal_price(side)
            if self.order_history:
                belief_prev_order = self.belief_function(self.order_history["price"], self.order_history["side"], self.get_order_list()[0])
                surplus_prev_order = self.order_history["side"]*(estimate + self.pv.value_for_exchange(self.position, self.order_history["side"]) - self.order_history["price"])
                if opt_price_est_surplus < belief_prev_order * surplus_prev_order:
                    order = Order(
                        price=self.order_history["price"],
                        quantity=1,
                        agent_id=self.get_id(),
                        time=t,
                        order_type=self.order_history["side"],
                        order_id=random.randint(1, 10000000)
                    )
                    self.order_history = {"id": order.order_id, "side":self.order_history["side"], "price":order.price, "transacted": False}
    
                    # print("Early exit", timer.time() - start_time)
                    # input()
                    return [order]
            
            order = Order(
                price=opt_price,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=side,
                order_id=random.randint(1, 10000000)
            )
            self.order_history = {"id": order.order_id, "side":side, "price":order.price, "transacted": False}
            
            # print("HBL NORMAL", timer.time() - start_time)
            # input()
            return [order]

        else:
            # ZI Agent
            valuation_offset = spread*random.random() + self.shade[0]
            if side == BUY:
                price = estimate + self.pv.value_for_exchange(self.position, BUY) - valuation_offset
            elif side == SELL:
                price = estimate + self.pv.value_for_exchange(self.position, SELL) + valuation_offset
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

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)
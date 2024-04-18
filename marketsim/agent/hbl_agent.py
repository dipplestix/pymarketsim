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
from fastcubicspline import NPointPoly

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

            for ind, order in enumerate(orders):
                found_matched = False
                for matched_order in self.market.matched_orders:
                    if order.order_id == matched_order.order.order_id:
                        if matched_order.order.order_type == BUY and matched_order.price <= p:
                            TBL += 1
                        found_matched = True
                        break
                if not found_matched:
                    if order.order_type == BUY and order.price >= p:
                        #order time to withdrawal time
                        withdrawn = False
                        latest_order_time = 0
                        for i in range(ind + 1, len(orders)):
                            if orders[i].agent_id == order.agent_id and orders[i].order_id != order.order_id:
                                latest_order_time = orders[i].time
                                withdrawn = True
                                break
                        if not withdrawn:
                            #Order not withdrawn
                            alive_time = current_time - order.time
                            if alive_time >= self.grace_period:
                                #Rejected
                                RBG += 1
                            else:
                                #Partial rejection
                                RBG += (alive_time / self.grace_period)
                        else:
                            #Withdrawal
                            # print(orders)
                            # print("ORDER TIME", order.time)
                            # print(self.market.get_time())
                            # input(latest_order_time)
                            time_till_withdrawal = latest_order_time - order.time
                            #Withdrawal
                            if time_till_withdrawal >= self.grace_period:
                                RBG += 1
                            else:
                                RBG += time_till_withdrawal / self.grace_period
                        
            #print("BUY BELIEF TIME", t.time() - start_time)
            #print("BY", p, TBL, AL, RBG)
            #input()
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

            for ind, order in enumerate(orders):
                found_matched = False
                for matched_order in self.market.matched_orders:
                    if order.order_id == matched_order.order.order_id:
                        if matched_order.order.order_type == SELL and matched_order.price >= p:
                            TAG += 1
                        found_matched = True
                        break
                if not found_matched:
                    if order.order_type == SELL and order.price <= p:
                        #order time to withdrawal time
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
                                RAL += 1
                            else:
                                RAL += alive_time / self.grace_period
                        else:
                            time_till_withdrawal = latest_order_time - order.time
                            if time_till_withdrawal >= self.grace_period:
                                RAL += 1
                            else:
                                RAL += time_till_withdrawal / self.grace_period
            #print("SELL BELIEF TIME", t.time() - start_time)
            # print(p, TAG, BG, RAL)
            # input()
            if TAG + BG == 0:
                return 0
            else:
                return (TAG + BG) / (TAG + BG + RAL)
    
    def get_order_list(self):
        lower_bound_mem = self.get_last_trade_time_step()

        buy_orders_memory = []
        sell_orders_memory = []
        last_L_orders = []
        for time in range(lower_bound_mem, self.market.get_time() + 1):
            for order in self.market.event_queue.scheduled_activities[time]:
                last_L_orders.append(order)
                if order.order_type == BUY:
                    buy_orders_memory.append(order)
                else:
                    sell_orders_memory.append(order)
        return last_L_orders, buy_orders_memory, sell_orders_memory

    def determine_optimal_price(self, side):
        '''
            Reference: https://www.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2009/notes/joel.pdf
            http://spider.sci.brooklyn.cuny.edu/~parsons/courses/840-spring-2005/notes/das.pdf 

            Spline time is ~__% of the run time. ___/___
        '''
        #start_time = t.time()
        last_L_orders, buy_orders_memory, sell_orders_memory = self.get_order_list()
        buy_orders_memory = sorted(buy_orders_memory, key = lambda order:order.price)
        sell_orders_memory = sorted(sell_orders_memory, key = lambda order:order.price)
        best_ask = float(self.market.order_book.sell_unmatched.peek())
        best_buy = float(self.market.order_book.buy_unmatched.peek())
        if side == BUY: 
            best_buy_belief = self.belief_function(best_buy, BUY, last_L_orders)
            best_ask_belief = 1
            def interpolate(bound1, bound2, bound1Belief, bound2Belief):
                start_time = timer.time()
                #cs = sp.interpolate.CubicSpline([bound1, bound2], [bound1Belief, bound2Belief], extrapolate=False)
                cs = NPointPoly([bound1, bound2], [bound1Belief, bound2Belief])
                print("INTERPOLATE TIME", timer.time() - start_time)
                def optimize(price): return -((self.estimate_fundamental() + self.pv.value_for_exchange(self.position, BUY) - price) * cs(price))
                start_time = timer.time()
                max_x = sp.optimize.differential_evolution(optimize, [[bound1, bound2]])
                print("MAX TIME", timer.time() - start_time)
                return max_x.x.item(), -max_x.fun

            buy_high = float(buy_orders_memory[-1].price)
            buy_high_belief = self.belief_function(buy_high, BUY, last_L_orders)
            i = len(buy_orders_memory) - 1
            while i > 0 and self.belief_function(buy_orders_memory[i].price, BUY, last_L_orders) != 0:
                i -= 1
            buy_low = buy_orders_memory[i].price
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
                    max_val = interpolate(buy_high, best_ask, buy_high_belief, 1)
                    optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                if best_buy >= buy_low:
                    if best_buy != buy_high:
                        #interpolate between best buy and buy_high 
                        max_val = interpolate(best_buy, buy_high, best_buy_belief, buy_high_belief)
                        optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                    if best_buy != buy_low:
                        #interpolate between best buy and buy_low
                        max_val_2 = interpolate(buy_low, best_buy, buy_low_belief, best_buy_belief)
                        optimal_price = max(optimal_price, max_val_2, key=lambda pair: pair[1])
                    #interpolate between buy_low and 0
                    if buy_low_belief > 0:
                        lower_bound = buy_low - 2 * (best_ask - buy_low)
                        max_val_3 = interpolate(lower_bound, buy_low, 0, buy_low_belief)
                        optimal_price = max(optimal_price, max_val_3, key=lambda pair: pair[1])
                        
                elif best_buy < buy_low:
                    #interpolate between buy_high and buy_low
                    if buy_high != buy_low:
                        max_val = interpolate(buy_low, buy_high, buy_low_belief, buy_high_belief)
                        optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                    #interpolate buy_low and best_buy
                    if buy_low != best_buy:
                        max_val_2 = interpolate(best_buy, buy_low, best_buy_belief, buy_low_belief)
                        optimal_price = max(optimal_price, max_val_2, key=lambda pair: pair[1])
                    #interpolate best_buy and 0?
                    if best_buy_belief > 0:
                        lower_bound = best_buy - 2 * (best_ask - best_buy)
                        max_val_3 = interpolate(lower_bound, best_buy, 0, best_buy_belief)
                        optimal_price = max(optimal_price, max_val_3, key=lambda pair: pair[1])

            elif buy_high < best_buy:
                # interpolate between best_ask and best_buy
                # occasionally have bug where best buy is > best_ask? see slurm-6885793
                if best_ask != best_buy:
                    try:
                        max_val = interpolate(best_buy, best_ask, best_buy_belief, best_ask_belief)
                        optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                    except:
                        print("ODD BUG TO HAVE...", self.market.order_book.buy_unmatched, self.market.order_book.sell_unmatched , best_buy, best_ask, best_buy_belief, best_ask_belief,
                        float(self.market.order_book.buy_unmatched.peek()), float(self.market.order_book.sell_unmatched.peek()), self.market.order_book.buy_unmatched.heap, self.market.order_book.sell_unmatched.heap)
                        pass
                #interpolate between best_buy and buy_high
                if best_buy != buy_high:
                    max_val_2 = interpolate(buy_high, best_buy, buy_high_belief, best_buy_belief)
                    optimal_price = max(optimal_price, max_val_2, key=lambda pair: pair[1])

                #interpolate between buy_high and buy_low
                if buy_high != buy_low:
                    max_val_3 = interpolate(buy_low, buy_high, buy_low_belief, buy_high_belief)
                    optimal_price = max(optimal_price, max_val_3, key=lambda pair: pair[1])

                #interpolate buy_low and 0
                if buy_low_belief > 0:
                    lower_bound = buy_low - 2 * (best_ask - buy_low)
                    max_val_3 = interpolate(lower_bound, buy_low, 0, buy_low_belief)
                    optimal_price = max(optimal_price, max_val_3, key=lambda pair: pair[1])
                    
            if optimal_price == (0,0):
                print("BUY", buy_low, buy_low_belief, buy_high, buy_high_belief, best_buy, best_buy_belief, best_ask)
                input("ERROR")
            #Adjusting for multiple 0 values (from the function). Edge case in case order with belief = 0 transacts.
            if optimal_price[0] > self.estimate_fundamental() + self.pv.value_for_exchange(self.position, BUY):
                # print("ORIGINAL and SURPLUS", optimal_price[0], optimal_price[1], (self.estimate_fundamental() + self.pv.value_for_exchange(self.position, BUY) - optimal_price[0]))
                # print("Adjusted belief", self.belief_function(self.estimate_fundamental() + self.pv.value_for_exchange(self.position, side), BUY, last_L_orders))
                # print(self.estimate_fundamental() + self.pv.value_for_exchange(self.position, BUY))
                return self.estimate_fundamental() + self.pv.value_for_exchange(self.position, BUY), 0
            # print("ESTIMATED BUY EXPECTED SURPLUS VALUE OF OPT", optimal_price[1])
            # print("BUY BELIEF", self.belief_function(optimal_price[0], BUY, last_L_orders))
            return optimal_price[0], optimal_price[1]

        else:
            best_buy_belief = 1
            best_ask_belief = self.belief_function(best_ask, SELL, last_L_orders)
            i = 0
            while i < len(sell_orders_memory) - 1 and self.belief_function(sell_orders_memory[i].price, SELL, last_L_orders) != 0:
                i += 1
            sell_high = sell_orders_memory[i].price
            sell_high_belief = self.belief_function(sell_orders_memory[i].price, SELL, last_L_orders)
            
            optimal_price = (0,-sys.maxsize)
            best_buy_belief = 1
            sell_low = float(sell_orders_memory[0].price)
            sell_low_belief = self.belief_function(sell_low, SELL, last_L_orders)
            # print("SELL", sell_low, sell_low_belief, sell_high, sell_high_belief, best_ask, best_buy)
            # input()
            def interpolate(bound1, bound2, bound1Belief, bound2Belief):
                start_time = timer.time()
                #cs = sp.interpolate.CubicSpline([bound1, bound2], [bound1Belief, bound2Belief], extrapolate=False)
                cs = NPointPoly([bound1, bound2], [bound1Belief, bound2Belief])
                print("SELL INTERPOLATE TIME", timer.time() - start_time)
                def optimize(price): return -((price - (self.estimate_fundamental() + self.pv.value_for_exchange(self.position, SELL))) * cs(price))
                start_time = timer.time()
                max_x = sp.optimize.differential_evolution(optimize, bounds=[[bound1, bound2]])
                print("SELL MAX TIME", timer.time() - start_time)
                return max_x.x.item(), -max_x.fun

            if best_buy > sell_low:
                sell_low = best_buy
                sell_low_belief = 1
                sell_high = max(sell_high, sell_low)
                sell_high_belief = min(sell_high_belief, sell_low_belief)

            # print("SELL 2", sell_low, sell_low_belief, sell_high, sell_high_belief, best_ask, best_buy)
            # input()
            if sell_low <= best_ask:
                #interpolate best buy to sell_low
                if sell_low != best_buy:
                    max_val = interpolate(best_buy, sell_low, best_buy_belief, sell_low_belief)
                    optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                if best_ask <= sell_high:
                    if sell_low != best_ask:
                        #interpolate sell_low to best_ask
                        max_val = interpolate(sell_low, best_ask, sell_low_belief, best_ask_belief)
                        optimal_price = max(optimal_price, max_val, key=lambda pair:pair[1])
                    if best_ask != sell_high:
                        #interpolate best_ask to sell_high
                        max_val_2 = interpolate(best_ask, sell_high, best_ask_belief, sell_high_belief)
                        optimal_price = max(optimal_price, max_val_2, key=lambda pair:pair[1])
        
                    # interpolate sell_high to ????
                    if sell_high_belief > 0:
                        upper_bound = sell_high + 2 * (sell_high - best_buy)
                        max_val_3 = interpolate(sell_high, upper_bound, sell_high_belief, 0)
                        optimal_price = max(optimal_price, max_val_3, key=lambda pair:pair[1])
        
                elif best_ask > sell_high:
                    if sell_low != sell_high:
                        #interpolate low sell to high sell
                        max_val = interpolate(sell_low, sell_high, sell_low_belief, sell_high_belief)
                        optimal_price = max(optimal_price, max_val, key=lambda pair:pair[1])
                    if sell_high != best_ask:
                        #interpolate sell_high to best ask
                        max_val_2 = interpolate(sell_high, best_ask, sell_high_belief, best_ask_belief)
                        optimal_price = max(optimal_price, max_val_2, key=lambda pair:pair[1])

                    #interpolate sell_high to sell_high + 2*spread
                    if best_ask_belief > 0:
                        upper_bound = best_ask + 2 * (best_ask - best_buy)
                        max_val_3 = interpolate(best_ask, upper_bound, best_ask_belief, 0)
                        optimal_price = max(optimal_price, max_val_3, key=lambda pair:pair[1])

            elif sell_low > best_ask:
                if best_buy != best_ask:
                    #interpolate best_buy to best_ask
                    try:
                        max_val = interpolate(best_buy, best_ask, best_buy_belief, best_ask_belief)
                        optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                    except:
                        print("ODD BUG TO HAVE IN SELL...", self.market.order_book.buy_unmatched, self.market.order_book.sell_unmatched , best_buy, best_ask, best_buy_belief, best_ask_belief,
                        float(self.market.order_book.buy_unmatched.peek()), float(self.market.order_book.sell_unmatched.peek()))
                        pass
                if best_ask != sell_low:
                    #interpolate best_ask to sell_low
                    max_val_2 = interpolate(best_ask, sell_low, best_ask_belief, sell_low_belief)
                    optimal_price = max(optimal_price, max_val_2, key=lambda pair:pair[1])
                if sell_low != sell_high:
                    #interpolate sell_low to sell_high
                    max_val_3 = interpolate(sell_low, sell_high, sell_low_belief, sell_high_belief)
                    optimal_price = max(optimal_price, max_val_3, key=lambda pair:pair[1])
                #interpolate sell_high to sell_high + 2*spread
                if sell_high_belief > 0:
                    upper_bound = sell_high + 2 * (sell_high - best_buy)
                    max_val_4 = interpolate(sell_high, upper_bound, sell_high_belief, 0)
                    optimal_price = max(optimal_price, max_val_4, key=lambda pair:pair[1])

            if optimal_price == (0,0):
                print("SELL", sell_low, sell_low_belief, sell_high, sell_high_belief, best_ask, best_buy)
                input("ERROR")
            #EDGE CASE
            if optimal_price[0] < self.estimate_fundamental() + self.pv.value_for_exchange(self.position, SELL):
                # print("ORIGINAL and SURPLUS", optimal_price[0], optimal_price[1], (optimal_price[0] - (self.estimate_fundamental() + self.pv.value_for_exchange(self.position, SELL))))
                # print("Adjusted belief", self.belief_function(self.estimate_fundamental() + self.pv.value_for_exchange(self.position, side), SELL, last_L_orders))
                # print(self.estimate_fundamental() + self.pv.value_for_exchange(self.position, side))
                return self.estimate_fundamental() + self.pv.value_for_exchange(self.position, SELL), 0
            # print("SELL EXPECTED SURPLUS VALUE OF OPT", optimal_price[1])
            # print("SELL BELIEF", self.belief_function(optimal_price[0], SELL, last_L_orders))
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
                # print(belief_prev_order)
                # print("CHECK", self.pv.value_for_exchange(self.position, self.order_history[-1]["side"]), self.order_history[-1]["price"])
                # print("PREV ORDER SURPLUS", belief_prev_order, surplus_prev_order, self.pv.value_for_exchange(self.position, self.order_history[-1]["side"]))
                # print("CHECK HERE", surplus, belief * surplus, belief_prev_order * surplus_prev_order)
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
                    # privateBenefit = self.pv.value_for_exchange(self.position, side)
                    # privateVal = self.estimate_fundamental() + privateBenefit
                    # print("BELIEF CHECK", belief)
                    # print(order)
                    # print(self.pv.value_for_exchange(self.position, side))
                    # print("PRIVATEVAL", privateVal)
                    # print("FUNDAMENTAL", self.estimate_fundamental())
                    # print("TOP BUY", self.market.order_book.buy_unmatched.peek())
                    # print("TOP SELL", self.market.order_book.sell_unmatched.peek())
                    # print("Current value", self.get_pos_value() + self.position*estimate + self.cash, self.get_pos_value(), self.cash, self.position)
                    # print("ORDER HISTORY", self.order_history)
                    # input("Order submitted")
                    # print("\n\n")
                    print("Early exit", timer.time() - start_time)
                    input()
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
            # privateBenefit = self.pv.value_for_exchange(self.position, side)
            # privateVal = self.estimate_fundamental() + privateBenefit
            # print("OPT", opt_price)
            # print(order)
            # print(self.pv.value_for_exchange(self.position, side))
            # print("PRIVATEVAL", privateVal)
            # print("FUNDAMENTAL", self.estimate_fundamental())
            # print("TOP BUY", self.market.order_book.buy_unmatched.peek())
            # print("TOP SELL", self.market.order_book.sell_unmatched.peek())
            # print("Current value", self.get_pos_value() + self.position*estimate + self.cash, self.get_pos_value(), self.cash, self.position)
            # print("ORDER HISTORY", self.order_history)
            # input("Order submitted")
            # print("\n\n")
            print("HBL NORMAL", timer.time() - start_time)
            input()
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
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

EPSILON = 1e-5

class HBLAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: List, L: int, pv_var: float, arrival_rate : float):
        self.agent_id = agent_id
        self.market = market
        self.pv = PrivateValues(q_max, pv_var)
        self.position = 0
        self.shade = shade
        self.cash = 0
        self.L = L
        self.ORDERS = 0
        self.HBL_MOVES = 0
        self.COUNTER = 0
        self.grace_period = 1/arrival_rate

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
                        if matched_order.order.order_type == BUY and matched_order.order.price <= p:
                            TBL += 1
                        found_matched = True
                        break
                if not found_matched:
                    if order.order_type == BUY and order.price >= p:
                        #order time to withdrawal time
                        withdrawn = False
                        latest_order_time = 0
                        for i in range(ind + 1, len(orders)):
                            if orders[i].agent_id == order.agent_id:
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
                                RBG += alive_time / self.grace_period
                        else:
                            #Withdrawal
                            time_till_withdrawal = latest_order_time - order.time
                            #Withdrawal
                            if time_till_withdrawal >= self.grace_period:
                                RBG += 1
                            else:
                                RBG += time_till_withdrawal / self.grace_period
                        
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
            for ind, order in enumerate(orders):
                found_matched = False
                for matched_order in self.market.matched_orders:
                    if order.order_id == matched_order.order.order_id:
                        if matched_order.order.order_type == SELL and matched_order.order.price >= p:
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
                            if alive_time > self.grace_period:
                                RAL += 1
                            else:
                                RAL += alive_time / self.grace_period
                        else:
                            # input("withdrawn")
                            # #Withdrawal
                            # print(time_till_withdrawal)
                            # print(current_time - order.time)
                            time_till_withdrawal = latest_order_time - order.time
                            if time_till_withdrawal >= self.grace_period:
                                RAL += 1
                            else:
                                RAL += time_till_withdrawal / self.grace_period
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
        buy_orders_memory = sorted(buy_orders_memory, key = lambda order:order.price)
        sell_orders_memory = sorted(sell_orders_memory, key = lambda order:order.price)
        best_ask = float(self.market.order_book.sell_unmatched.peek())
        best_buy = float(self.market.order_book.buy_unmatched.peek())
        if side == BUY: 
            best_buy_belief = self.belief_function(best_buy, BUY, last_L_orders)
            best_ask_belief = 1
            def interpolate(bound1, bound2, bound1Belief, bound2Belief):
                cs = sp.interpolate.CubicSpline([bound1, bound2], [bound1Belief, bound2Belief], extrapolate=False)
                
                # fig,ax = plt.subplots()
                # increment = (bound2 - bound1) / 100
                # vals = [bound1]
                # cs_vals = [cs(bound1)]
                # a = bound1
                # while a < bound2:
                #     vals.append(a)
                #     cs_vals.append(cs(a))
                #     a += increment
                # plt.plot(vals,cs_vals)
                # ax.set_xbound(vals[0], vals[-1])
                # ax.set_ybound(cs_vals[0], cs_vals[-1])
                # print(vals, cs_vals)
                # plt.show()

                #def optimize(price): return -((self.estimate_fundamental() 
                #    + self.pv.value_for_exchange(self.position, BUY) - price) * cs(price))
                
                def optimize(price): return -((self.estimate_fundamental() 
                    + self.pv.value_for_exchange(self.position, BUY) - price) * self.belief_function(price, BUY, last_L_orders))
                
                try:
                    max_x = sp.optimize.minimize_scalar(optimize, bounds=(
                        bound1, bound2), method='bounded')
                except Exception as inst:
                    print(bound1, bound2, bound1Belief, bound2Belief)
                    input(inst)
                #Flip max_x.fun again to make positive
                return max_x.x.item(), -max_x.fun

            def interpolate_end(bound1, bound2, bound1Belief, bound2Belief):
                slope = (bound2Belief - bound1Belief) / (bound2 - bound1)
                if slope == 0:
                    return (0, -sys.maxsize)
                root = (-bound2Belief / slope) + bound1
                def optimize(price): return -((self.estimate_fundamental() 
                    + self.pv.value_for_exchange(self.position, BUY) - price) * (slope * (price - root)))
                max_x = sp.optimize.minimize_scalar(optimize, bounds=(
                    root, bound2), method='bounded')
                return max_x.x.item(), -max_x.fun
            
            try:
                buy_high = float(buy_orders_memory[-1].price)
            except Exception as inst:
                print(buy_orders_memory)
                print(last_L_orders)
                print(self.market.matched_orders)
                input(inst)   
            buy_high_belief = self.belief_function(buy_high, BUY, last_L_orders)
            buy_low = float(buy_orders_memory[0].price)
            buy_low_belief = self.belief_function(buy_orders_memory[0].price, BUY, last_L_orders)
            # print("BUY LOW")
            # input(buy_low_belief)
            for i in range(len(buy_orders_memory) - 1, -1, -1):
                belief = self.belief_function(buy_orders_memory[i].price, BUY, last_L_orders)
                if belief == 0:
                    buy_low = float(buy_orders_memory[i].price)
                    buy_low_belief = belief
                    break                
            optimal_price = (0,-sys.maxsize)
            # print("BUY", buy_low, buy_low_belief, buy_high, buy_high_belief, best_buy, best_buy_belief, best_ask, best_ask_belief)
            # input()
            if buy_high >= best_ask:
                buy_high = best_ask
                buy_high_belief = best_ask_belief
                buy_low = min(buy_high, buy_low)
                buy_low_belief = min(buy_high_belief, buy_low_belief)
            # print("BUY2", buy_low, buy_low_belief, buy_high, buy_high_belief, best_buy, best_buy_belief, best_ask)
            # input()
            #Best ask > buy high
            if buy_high >= best_buy:
                #interpolate between best ask and buy high
                if best_ask != buy_high:
                    max_val = interpolate(buy_high, best_ask, buy_high_belief, 1)
                    # print("Enter 1", max_val.x.item())
                    # input()
                    optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                if best_buy >= buy_low:
                    if best_buy != buy_high:
                        #interpolate between best buy and buy_high 
                        max_val = interpolate(best_buy, buy_high, best_buy_belief, buy_high_belief)
                        # print("Enter 2", max_val)
                        # input()
                        optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                    if best_buy != buy_low:
                        #interpolate between best buy and buy_low
                        max_val_2 = interpolate(buy_low, best_buy, buy_low_belief, best_buy_belief)
                        # print("Enter 3", max_val_2)
                        # input()
                        optimal_price = max(optimal_price, max_val_2, key=lambda pair: pair[1])
                    #interpolate between buy_low and 0
                    # if buy_low_belief > 0:
                    #     max_val_3 = interpolate_end(buy_low, best_ask, buy_low_belief, best_ask_belief)
                    #     # print("Enter 4", max_val_3)
                    #     # input()
                    #     opt = optimal_price
                    #     optimal_price = max(optimal_price, max_val_3, key=lambda pair: pair[1])
                    #     if opt[0] != optimal_price[0]:
                    #         self.COUNTER += 1
                        
                elif best_buy < buy_low:
                    #interpolate between buy_high and buy_low
                    if buy_high != buy_low:
                        max_val = interpolate(buy_low, buy_high, buy_low_belief, buy_high_belief)
                        # print("Enter 5", max_val)
                        # input()
                        optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                    #interpolate buy_low and best_buy
                    if buy_low != best_buy:
                        max_val_2 = interpolate(best_buy, buy_low, best_buy_belief, buy_low_belief)
                        # print("Enter 6", max_val_2)
                        # input()
                        optimal_price = max(optimal_price, max_val_2, key=lambda pair: pair[1])
                    #interpolate best_buy and 0?
                    # if best_buy_belief > 0:
                    #     max_val_3 = interpolate_end(best_buy, best_ask, best_buy_belief, best_ask_belief)
                    #     # print("Enter 7", max_val_3)
                    #     # input()
                    #     opt = optimal_price
                    #     optimal_price = max(optimal_price, max_val_3, key=lambda pair: pair[1])
                    #     if opt[0] != optimal_price[0]:
                    #         self.COUNTER += 1

            elif buy_high < best_buy:
                #interpolate between best_ask and best_buy
                if best_ask != best_buy:
                    max_val = interpolate(best_buy, best_ask, best_buy_belief, best_ask_belief)
                    # print("Enter 8", max_val)
                    # input()
                    optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                #interpolate between best_buy and buy_high
                if best_buy != buy_high:
                    max_val_2 = interpolate(buy_high, best_buy, buy_high_belief, best_buy_belief)
                    # print("Enter 9", max_val_2)
                    # input()
                    optimal_price = max(optimal_price, max_val_2, key=lambda pair: pair[1])

                #interpolate between buy_high and buy_low
                if buy_high != buy_low:
                    max_val_3 = interpolate(buy_low, buy_high, buy_low_belief, buy_high_belief)
                    # print("Enter 10", max_val_3)
                    # input()
                    optimal_price = max(optimal_price, max_val_3, key=lambda pair: pair[1])

                #interpolate buy_low and 0
                # if buy_low_belief > 0: 
                #     max_val_4 = interpolate_end(buy_low, best_ask, buy_low_belief, best_ask_belief)
                #     # print("Enter 11", max_val_4)
                #     # input()
                #     opt = optimal_price
                #     optimal_price = max(optimal_price, max_val_4, key=lambda pair: pair[1])
                #     if opt[0] != optimal_price[0]:
                #         self.COUNTER += 1
            
            if optimal_price == (0,0):
                print("BUY2", buy_low, buy_low_belief, buy_high, buy_high_belief, best_buy, best_buy_belief, best_ask)
                input("ERROR")
            # input(optimal_price)
            return optimal_price[0]
        else:
            best_buy_belief = 1
            best_ask_belief = self.belief_function(best_ask, SELL, last_L_orders)
            try:
                sell_high = float(sell_orders_memory[-1].price)
            except Exception as inst:
                print(sell_orders_memory)
                print(last_L_orders)
                print(self.market.matched_orders)
                input(inst)
            sell_high_belief = self.belief_function(sell_high, SELL, last_L_orders)
            # input(sell_high_belief)
            for i in range(len(sell_orders_memory)):
                belief = self.belief_function(sell_orders_memory[i].price, SELL, last_L_orders)
                if belief == 0:
                    sell_high = float(sell_orders_memory[i].price)
                    sell_high_belief = belief
                    break                
            optimal_price = (0,-sys.maxsize)
            best_buy_belief = 1
            sell_low = float(sell_orders_memory[0].price)
            sell_low_belief = self.belief_function(sell_low, SELL, last_L_orders)
            # print("SELL", sell_low, sell_low_belief, sell_high, sell_high_belief, best_ask, best_buy)
            # input()
            def interpolate(bound1, bound2, bound1Belief, bound2Belief):
                cs = sp.interpolate.CubicSpline([bound1, bound2], [bound1Belief, bound2Belief], extrapolate=False)
                
                # fig,ax = plt.subplots()
                # increment = (bound2 - bound1) / 100
                # vals = [bound1]
                # cs_vals = [cs(bound1)]
                # a = bound1
                # while a < bound2:
                #     vals.append(a)
                #     cs_vals.append(cs(a))
                #     a += increment
                # plt.plot(vals,cs_vals)
                # ax.set_xbound(vals[0], vals[-1])
                # ax.set_ybound(cs_vals[0], cs_vals[-1])
                # print(vals, cs_vals)
                # plt.show()

                #def optimize(price): return -((
                #    price + self.pv.value_for_exchange(self.position, SELL) - self.estimate_fundamental()) * cs(price))
                
                def optimize(price): return -((self.estimate_fundamental() 
                    + self.pv.value_for_exchange(self.position, BUY) - price) * self.belief_function(price, SELL, last_L_orders))
                try:
                    max_x = sp.optimize.minimize_scalar(optimize, bounds=(
                        bound1, bound2), method='bounded')
                except Exception as inst:
                    print("SELL", bound1, bound2, bound1Belief, bound2Belief)
                    input(inst)
                return (max_x.x.item(), max_x.fun)

            def interpolate_end(bound1, bound2, bound1Belief, bound2Belief):
                slope = (bound2Belief - bound1Belief) / (bound2 - bound1)
                if slope == 0:
                    return (0, -sys.maxsize)
                root = (-bound2Belief / slope) + bound1
                def optimize(price): return -((
                    price + self.pv.value_at_position(self.position) - self.estimate_fundamental()) * (slope * (price - bound1) + bound1Belief))
                max_x = sp.optimize.minimize_scalar(optimize, bounds=(
                    bound1, root), method='bounded')
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
                    # print("Enter 1", max_val)
                    # input()
                    optimal_price = max(optimal_price, max_val, key=lambda pair: pair[1])
                if best_ask <= sell_high:
                    if sell_low != best_ask:
                        #interpolate sell_low to best_ask
                        max_val = interpolate(sell_low, best_ask, sell_low_belief, best_ask_belief)
                        # print("Enter 2", max_val)
                        # input()
                        optimal_price = max(optimal_price, max_val, key=lambda pair:pair[1])
                    if best_ask != sell_high:
                        #interpolate best_ask to sell_high
                        max_val_2 = interpolate(best_ask, sell_high, best_ask_belief, sell_high_belief)
                        # print("Enter 3", max_val_2)
                        # input()
                        optimal_price = max(optimal_price, max_val_2, key=lambda pair:pair[1])
        
                    #interpolate sell_high to ????
                    # if sell_high_belief > 0:
                    #     max_val_3 = interpolate_end(best_buy, sell_high, best_buy_belief, sell_high_belief)
                    #     # print("Enter 4", max_val_3)
                    #     # input()
                    #     opt = optimal_price
                    #     optimal_price = max(optimal_price, max_val_3, key=lambda pair:pair[1])
                    #     if opt[0] != optimal_price[0]:
                    #         self.COUNTER += 1
        
                elif best_ask > sell_high:
                    if sell_low != sell_high:
                        #interpolate low sell to high sell
                        max_val = interpolate(sell_low, sell_high, sell_low_belief, sell_high_belief)
                        # print("Enter 5", max_val)
                        # input()
                        optimal_price = max(optimal_price, max_val, key=lambda pair:pair[1])
                    if sell_high != best_ask:
                        #interpolate sell_high to best ask
                        max_val_2 = interpolate(sell_high, best_ask, sell_high_belief, best_ask_belief)
                        # print("Enter 6", max_val_2)
                        # input()                        
                        optimal_price = max(optimal_price, max_val_2, key=lambda pair:pair[1])

                    #interpolate sell_high to ????
                    # if sell_high_belief > 0:
                    #     max_val_3 = interpolate_end(best_buy, best_ask_belief, best_buy_belief, best_ask_belief)
                    #     # print("Enter 7", max_val_3)
                    #     # input()
                    #     opt = optimal_price
                    #     optimal_price = max(optimal_price, max_val_3, key=lambda pair:pair[1])
                    #     if opt[0] != optimal_price[0]:
                    #         self.COUNTER += 1

            elif sell_low > best_ask:
                if best_buy != best_ask:
                    #interpolate best_buy to best_ask
                    max_val = interpolate(best_buy, best_ask, best_buy_belief, best_ask_belief)
                    # print("Enter 8", max_val)
                    # input()                  
                    optimal_price = max(optimal_price, max_val, key=lambda pair:pair[1])
                if best_ask != sell_low:
                    #interpolate best_ask to sell_low
                    max_val_2 = interpolate(best_ask, sell_low, best_ask_belief, sell_low_belief)
                    # print("Enter 9", max_val_2)
                    # input()      
                    optimal_price = max(optimal_price, max_val_2, key=lambda pair:pair[1])
                if sell_low != sell_high:
                    #interpolate sell_low to sell_high
                    max_val_3 = interpolate(sell_low, sell_high, sell_low_belief, sell_high_belief)
                    # print("Enter 10", max_val_3)
                    # input()      
                    optimal_price = max(optimal_price, max_val_3, key=lambda pair:pair[1])
                #interpolate sell_high to ???
                # if sell_high_belief > 0:
                #     max_val_4 = interpolate_end(best_buy_belief, sell_high, best_buy_belief, sell_high_belief)
                #     # print("Enter 4", max_val_4)
                #     # input()
                #     opt = optimal_price
                #     optimal_price = max(optimal_price, max_val_4, key=lambda pair:pair[1])
                #     if opt[0] != optimal_price[0]:
                #         self.COUNTER += 1
            if optimal_price == (0,0):
                print("SELL 2", sell_low, sell_low_belief, sell_high, sell_high_belief, best_ask, best_buy)
                input("ERROR")
            # input(optimal_price)
            return optimal_price[0]

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
            order = Order(
                price=opt_price,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=side,
                order_id=random.randint(1, 10000000)
            )
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
        return f'ZI{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)
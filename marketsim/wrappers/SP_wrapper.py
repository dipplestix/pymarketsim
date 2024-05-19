import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from fourheap.constants import BUY, SELL
from market.market import Market
from fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from fundamental.mean_reverting import GaussianMeanReverting
from agent.zero_intelligence_agent import ZIAgent
from agent.spoofer import SpoofingAgent
from agent.hbl_agent import HBLAgent

import torch.distributions as dist
import torch
from collections import defaultdict

def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps


'''
Helpful notes:

Ep_length: Number of times the Spoofer enters because step will run intermediate steps
lamSP: 5e-3, 5e-2 (~50-55 entries per 1000), 5e-1 (~500 entries)
'''
class SPEnv(gym.Env):
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_assets: int = 1,
                 lam: float = 5e-3,
                 lamSP: float = 5e-2, # Tune
                 mean: float = 1e5,
                 r: float = 0.05,
                 shock_var: float = 1e5,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 order_size=1, # the size of regular order: NEED TUNING
                 spoofing_size=200, # the size of spoofing order: NEED TUNING
                 normalizers = None, # normalizer for obs: NEED TUNING
                 pvalues = None,
                 sampled_arr = None,
                 fundamental = None
                 ):

        # MarketSim Setup
        if shade is None:
            shade = [10, 30]
        self.num_agents = num_background_agents
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.time = 0

        self.sampled_arr = sampled_arr

        self.most_recent_trade = {key: np.nan for key in range(0, sim_time + 1)}
        self.spoof_orders = {key: np.nan for key in range(0, sim_time + 1)}
        self.sell_orders = {key: np.nan for key in range(0, sim_time + 1)}
        self.best_buys = {key: np.nan for key in range(0, sim_time + 1)}
        self.best_asks = {key: np.nan for key in range(0, sim_time + 1)}
        self.count = 0

        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = self.sim_time
        # self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
        self.arrival_times = sampled_arr
        self.arrival_index = 0

        # Spoofer
        self.lamSP = lamSP
        self.arrivals_SP = defaultdict(list)
        self.arrival_times_SP = sample_arrivals(lamSP, self.arrivals_sampled)
        self.arrival_index_SP = 0
        self.normalizers = normalizers

        # Set up markets
        self.markets = []
        for _ in range(num_assets):
            fundamental = fundamental
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        # Set up for regular traders.
        self.agents = {}
        for agent_id in range(12):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1
            self.agents[agent_id] = (
                ZIAgent(
                    agent_id=agent_id,
                    market=self.markets[0],
                    q_max=q_max,
                    shade=shade,
                    pv_var=pv_var,
                    pv=pvalues[agent_id]
                ))

        for agent_id in range(12,25):
                self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
                self.arrival_index += 1
                self.agents[agent_id] = (HBLAgent(
                    agent_id = agent_id,
                    market = self.markets[0],
                    pv_var = pv_var,
                    q_max= q_max,
                    shade = shade,
                    L = 4,
                    arrival_rate = self.lam,
                    pv=pvalues[agent_id]
                ))

        # Set up for spoofer.
        self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1000].append(self.num_agents)
        self.arrival_index_SP += 1
        # print(self.arrival_times_SP,self.arrivals_SP)
        # print(self.arrival_times,self.arrivals)
        # input()
        self.spoofer = SpoofingAgent(
            agent_id=self.num_agents,
            market=self.markets[0],
            q_max=q_max,
            pv_var=pv_var,
            order_size=order_size,
            spoofing_size=spoofing_size,
            normalizers=normalizers
        )

        self.means = {key: [] for key in range(0, 10001)}
        self.bestSells = {key: [] for key in range(0, 10001)}
        self.bestBids = {key: [] for key in range(0, 10001)}

        # Gym Setup
        """
        Given a market state s when the self agent arrives at time t, 
        the agent receives an observation O(s) that includes the number of time steps left T - t, 
        the current fundamental value rt, 
        the current best BID and ASK price in the limit order book (if any), 
        the self agent's inventory I, 
        the self agent's cash,
        """
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, -1, -10]), high=np.array([1.0, 1.0, 1.0, 1.0, 1, 10]), shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32) # price for regular order and price for spoofing

    def get_obs(self):
        return self.observation

    def update_obs(self): # TODO: Check if this observation works.
        self.time_left = self.sim_time - self.time
        self.fundamental_value = self.markets[0].fundamental.get_value_at(self.time)
        self.best_ask = self.markets[0].order_book.sell_unmatched.peek()
        self.best_bid = self.markets[0].order_book.buy_unmatched.peek()
        self.SPinvt = self.spoofer.position
        self.SPcash = self.spoofer.cash

        self.observation = self.normalization(
            time_left=self.time_left,
            fundamental_value=self.fundamental_value,
            best_ask=self.best_ask,
            best_bid=self.best_bid,
            SPinvt=self.SPinvt,
            SPcash=self.SPcash)

    def normalization(self,
                      time_left: int,
                      fundamental_value: float,
                      best_ask: float,
                      best_bid: float,
                      SPinvt: float,
                      SPcash: float):
        '''
            We need to define min/max bounds for the price or else the range is TOO BIG.
            Can't have a fundamental of 1e5 and a spoofing order of price = 7 for example. 
        '''
        if self.normalizers is None:
            print("No normalizer warning!")
            return np.array([time_left, fundamental_value, best_ask, best_bid, SPinvt, SPcash])

        time_left /= self.sim_time
        #TODO
        fundamental_value /= self.normalizers["fundamental"]
        # print("BEST ASK")
        # print(best_ask, best_bid)
        #TODO: fundamental_value OR self.normalizers["fundamental"]?
        # self.normalizers["min_order_val"] = abs(self.markets[0].order_book.get_low_bid())
        # self.normalizers["order_price"] = self.markets[0].order_book.get_high_ask() - self.normalizers["min_order_val"] 
        # print("BEST ASK")
        # input(best_ask)
        # print("BEST BID")
        # print(self.normalizers["spoofing"])
        # input(best_bid)
        if math.isinf(abs(best_ask)):
            best_ask = 1
        else:
            best_ask = (best_ask - 5e4) / (2e5-5e4)

        if math.isinf(abs(best_bid)):
            best_bid = 0
        else:
            best_bid = (best_bid - 5e4) / (2e5-5e4)


        SPinvt /= self.normalizers["invt"]
        SPcash /= self.normalizers["cash"]

        return np.array([time_left, fundamental_value, best_ask, best_bid, SPinvt, SPcash])

    def reset(self, seed=None, options=None):
        self.time = 0
        self.observation = None

        # Reset the markets
        for market in self.markets:
            market.reset()

        # Reset the agents
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.reset()

        # Reset spoofer
        self.spoofer.reset()
        self.update_obs()

        # Reset Arrivals
        self.reset_arrivals()

        self.most_recent_trade = {key: np.nan for key in range(0, self.sim_time + 1)}
        self.spoof_orders = {key: np.nan for key in range(0, self.sim_time + 1)}
        self.sell_orders = {key: np.nan for key in range(0, self.sim_time + 1)}
        self.best_buys = {key: np.nan for key in range(0, self.sim_time + 1)}
        self.best_asks = {key: np.nan for key in range(0, self.sim_time + 1)}

        # Run until the spoofer enters.
        # _ = self.run_until_next_SP_arrival()
        # self.run_agents_only()

        return self.get_obs(), {}

    def reset_arrivals(self):
        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = self.sim_time
        # self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
        self.arrival_times = self.sampled_arr
        self.arrival_index = 0

        self.arrivals_SP = defaultdict(list)
        self.arrival_times_SP = sample_arrivals(self.lamSP, self.arrivals_sampled)
        self.arrival_index_SP = 0

        for agent_id in range(self.num_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

        #Xintong paper - Spoofer arrives after timestep 1000
        self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1000].append(self.num_agents)
        self.arrival_index_SP += 1

    def step(self, action):
        if self.time < self.sim_time:
            # Only matters for first iteration through.
            if len(self.arrivals_SP[self.time]) != 0:
                reward = self.SP_step(action)
            else:
                reward = 0
                self.agents_step()
            self.time += 1
            end = self.run_until_next_SP_arrival()
            if end:
                return self.end_sim()
            return self.get_obs(), reward, False, False, {}
        else:
            return self.end_sim()

    def agents_step(self):
        agents = self.arrivals[self.time]
        if self.time < self.sim_time:
            for market in self.markets:
                market.event_queue.set_time(self.time)
                for agent_id in agents:
                    agent = self.agents[agent_id]
                    market.withdraw_all(agent_id)
                    side = random.choice([BUY, SELL])
                    orders = agent.take_action(side)
                    market.add_orders(orders)
                   
                    if self.arrival_index == self.arrivals_sampled:
                        self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
                        self.arrival_index = 0
                    self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + self.time].append(agent_id)
                    self.arrival_index += 1

                new_orders = market.step()
                # if not math.isinf(market.order_book.sell_unmatched.peek()) and not math.isinf(market.order_book.buy_unmatched.peek()):
                #     self.means[market.get_time()].append((market.order_book.sell_unmatched.peek() + market.order_book.buy_unmatched.peek()) / 2)
                #     self.bestBids[market.get_time()].append(market.order_book.buy_unmatched.peek())
                #     self.bestSells[market.get_time()].append(market.order_book.sell_unmatched.peek())

                for matched_order in new_orders:
                    agent_id = matched_order.order.agent_id
                    quantity = matched_order.order.order_type * matched_order.order.quantity
                    cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
                    if agent_id == self.num_agents:
                        self.spoofer.update_position(quantity, cash)
                    else:
                        self.agents[agent_id].update_position(quantity, cash)
                
                if not math.isinf(self.markets[0].order_book.sell_unmatched.peek()):
                    self.best_asks[self.time] = self.markets[0].order_book.sell_unmatched.peek()
                if not math.isinf(self.markets[0].order_book.buy_unmatched.peek()):
                    self.best_buys[self.time] = self.markets[0].order_book.buy_unmatched.peek() 
                if len(self.markets[0].matched_orders) > 0:
                    if self.time > 9900 and self.markets[0].matched_orders[-1].price != self.most_recent_trade[self.time - 1]:
                        a = self.markets[0].matched_orders[-1].price
                    self.most_recent_trade[self.time] = self.markets[0].matched_orders[-1].price

        else:
            self.end_sim()

    def SP_step(self, action):
        for market in self.markets:
            market.event_queue.set_time(self.time)
            market.withdraw_all(self.num_agents)
            orders = self.spoofer.take_action(action)
            # print(self.count)
            market.add_orders(orders)
            self.spoof_orders[self.time] = orders[1].price
            self.sell_orders[self.time] = orders[0].price
            if not math.isinf(self.markets[0].order_book.sell_unmatched.peek()):
                self.best_asks[self.time] = self.markets[0].order_book.sell_unmatched.peek()
            if not math.isinf(self.markets[0].order_book.buy_unmatched.peek()):
                self.best_buys[self.time] = self.markets[0].order_book.buy_unmatched.peek() 

            if self.arrival_index_SP == self.arrivals_sampled:
                self.arrival_times_SP = sample_arrivals(self.lamSP, self.arrivals_sampled)
                self.arrival_index_SP = 0
            self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1 + self.time].append(self.num_agents)
            # print(self.arrival_times_SP[self.arrival_index_SP].item() + 1 + self.time)
            self.arrival_index_SP += 1

            new_orders = market.step()
            for matched_order in new_orders:
                agent_id = matched_order.order.agent_id
                quantity = matched_order.order.order_type * matched_order.order.quantity
                cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
                if agent_id == self.num_agents:
                    self.spoofer.update_position(quantity, cash)
                else:
                    self.agents[agent_id].update_position(quantity, cash)

            estimated_fundamental = self.spoofer.estimate_fundamental()
            current_value = self.spoofer.position * estimated_fundamental + self.spoofer.cash
            current_value /= self.normalizers["reward"]
            reward = current_value - self.spoofer.last_value
            self.spoofer.last_value = reward #TODO: Check if we need to normalize the reward
            # if len(self.markets[0].matched_orders) > 0:
            #     if self.time > 9900 and self.markets[0].matched_orders[-1].price != self.most_recent_trade[self.time - 1]:
            #         a = self.markets[0].matched_orders[-1].price
            #     self.most_recent_trade[self.time] = self.markets[0].matched_orders[-1].price
        x = reward #TODO: Check if this normalizer works.
        # if abs(x) > 1:
        #     input(x)
        return x

    def end_sim_summarize(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        values = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            values[agent_id] = agent.get_pos_value() + agent.position * fundamental_val + agent.cash

        values[self.num_agents] = self.spoofer.position * fundamental_val + self.spoofer.cash
        # print(f'At the end of the simulation we get {values}')

    def end_sim(self):
        self.count += 1
        # print(self.count)
        # if self.count % 1 == 0:
        #     times = []
        #     means = []
        #     bidsTime = []
        #     bids = []
        #     sellsTime = []
        #     sells = []
        #     for key in self.means:
        #         if len(self.means[key]) != 0:
        #             times.append(key)
        #             means.append(np.mean(self.means[key]))
        #         if len(self.bestBids[key]) != 0:
        #             bidsTime.append(key)
        #             bids.append(np.mean(self.bestBids[key]))
        #         if len(self.bestSells[key]) != 0:
        #             sellsTime.append(key)
        #             sells.append(np.mean(self.bestSells[key]))
        #     preSpoof = []
        #     postSpoof = []
        #     sellPreSpoof = []
        #     sellPostSpoof = []
        #     agentCat = []
            # for agent in self.agents:
            #     if agent >= 6:
            #         print(self.agents[agent])
            #         agentCat.append(self.agents[agent])
            #         print(sum(1 for price in self.agents[agent].prices_before_spoofer if price >= 0))
            #         print(len(self.agents[agent].prices_before_spoofer))
            #         print(sum(1 for price in self.agents[agent].prices_after_spoofer if price >= 0))
            #         print(len(self.agents[agent].prices_after_spoofer))
            #         print(sum(price for price in self.agents[agent].prices_after_spoofer if price >= 0)/max(sum(1 for price in self.agents[agent].prices_after_spoofer if price >= 0),1))
            #         preSpoof.append(sum(1 for price in self.agents[agent].prices_before_spoofer if price >= 0) / max(len(self.agents[agent].prices_before_spoofer),1))
            #         postSpoof.append(sum(1 for price in self.agents[agent].prices_after_spoofer if price >= 0) / max(len(self.agents[agent].prices_after_spoofer),1))

            #         sellPreSpoof.append(sum(1 for price in self.agents[agent].sell_before_spoofer if price >= 0) / max(len(self.agents[agent].sell_before_spoofer),1))
            #         sellPostSpoof.append(sum(1 for price in self.agents[agent].sell_after_spoofer if price >= 0) / max(len(self.agents[agent].sell_after_spoofer),1))

            # bar_width = 0.23  # Width of each bar
            # x = np.arange(len(agentCat))  # The label locations
            # # Plot bars
            # plt.bar(x - bar_width/2, preSpoof, bar_width, label='PreSpoof')
            # plt.bar(x + bar_width/2, postSpoof, bar_width, label='PostSpoof')
            # # Add labels and title
            # plt.xlabel('HBL identifier')
            # plt.ylabel('Probability')
            # plt.title('Frequency that prespoof/postspoof orders > best buy')
            # plt.xticks(x, agentCat)  # Set category labels
            # plt.legend()  # Add legend

            # for agent in self.agents:
            #     if agent >= 6:
            #         print(self.agents[agent].buy_count)
            # # Show plot
            # print("\n\n")
            # plt.show()
            # bar_width = 0.23  # Width of each bar
            # x = np.arange(len(agentCat))  # The label locations
            # # Plot bars
            # plt.bar(x - bar_width/2, sellPreSpoof, bar_width, label='PreSpoof')
            # plt.bar(x + bar_width/2, sellPostSpoof, bar_width, label='PostSpoof')
            # # Add labels and title
            # plt.xlabel('HBL identifier')
            # plt.ylabel('Probability')
            # plt.title('Frequency that prespoof/postspoof orders > best ask')
            # plt.xticks(x, agentCat)  # Set category labels
            # plt.legend()  # Add legend

            # for agent in self.agents:
            #     if agent >= 6:
            #         print(self.agents[agent].sell_count)
            # # Show plot
            # plt.show()
                    
            # fig, axs = plt.subplots(2)
            # axs[0].plot(times, means, marker='o', linestyle='-')
            # axs[0].set_xlabel('Timesteps')
            # axs[0].set_ylabel('Price')
            # axs[0].set_title('Market Price of Stock')
            # axs[0].grid(True)
            
            # axs[1].plot(bidsTime, bids, marker='o', linestyle='-')
            # axs[1].plot(sellsTime, sells, marker='o', linestyle='-',color="red")
            # axs[1].set_xlabel('Timesteps')
            # axs[1].set_ylabel('Price')
            # axs[1].set_title('Bid-Ask Values')
            # axs[1].grid(True)
            # plt.show()

            
        estimated_fundamental = self.spoofer.estimate_fundamental()
        current_value = self.spoofer.position * estimated_fundamental + self.spoofer.cash
        reward = current_value - self.spoofer.last_value
        return self.get_obs(), reward / self.normalizers["reward"], True, False, {}


    def run_until_next_SP_arrival(self):
        while len(self.arrivals_SP[self.time]) == 0 and self.time < self.sim_time:
            self.agents_step()
            # print(self.markets[0].order_book.observe())
            self.time += 1

        if self.time >= self.sim_time:
            return True
        else:
            self.update_obs()
            return False

    def run_agents_only(self):
        for t in range(int(0.1 * self.sim_time)):
            if self.arrivals[t]:
                self.agents_step()
            self.time += 1

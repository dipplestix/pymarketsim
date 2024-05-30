import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from agent.paired_spoofer import SpooferZIAgent
from fourheap.order import Order
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
class NonSPEnv(gym.Env):
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
                 pvalues = None,
                 sampled_arr = None,
                 spoofer_arrivals = None,
                 fundamental = None,
                 analytics = False,
                 random_seed = None,
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
        self.spoofer_arrivals = spoofer_arrivals

        self.analytics = analytics
        
        if analytics:
            self.most_recent_trade = {key: np.nan for key in range(0, sim_time + 1)}
            self.best_buys = {key: np.nan for key in range(0, sim_time + 1)}
            self.best_asks = {key: np.nan for key in range(0, sim_time + 1)}
            self.sell_above_best = []
            self.spoofer_quantity = {key: np.nan for key in range(0, sim_time + 1)}
            self.spoofer_value = {key: np.nan for key in range(0, sim_time + 1)}
            self.trade_volume = {key: 0 for key in range(0, self.sim_time + 1)}
            self.mid_prices = {key: np.nan for key in range(0, self.sim_time + 1)}

        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = self.sim_time
        # self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
        self.arrival_times = sampled_arr
        self.arrival_index = 0

        self.random_seed = random_seed

        # Spoofer
        self.lamSP = lamSP
        self.arrivals_SP = defaultdict(list)
        if spoofer_arrivals != None:
            self.arrival_times_SP = spoofer_arrivals
        else:
            self.arrival_times_SP = sample_arrivals(lamSP, sim_time)
        self.arrival_index_SP = 0

        # Set up markets
        self.markets = []
        for _ in range(num_assets):
            fundamental = fundamental
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        # Set up for regular traders.
        self.agents = {}
        for agent_id in range(6):
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

        for agent_id in range(6, self.num_agents):
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
 
        self.spoofer = SpooferZIAgent(
                    agent_id=self.num_agents,
                    market=self.markets[0],
                    q_max=q_max,
                    shade=shade,
                    pv_var=pv_var,
                    pv=pvalues[self.num_agents]
                )        

        self.means = {key: [] for key in range(0, 10001)}
        self.bestSells = {key: [] for key in range(0, 10001)}
        self.bestBids = {key: [] for key in range(0, 10001)}

    def reset(self, seed=None, options=None):
        self.time = 0
      
        # Reset the markets
        for market in self.markets:
            market.reset()

        # Reset the agents
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.reset()

        # Reset spoofer
        self.spoofer.reset()

        # Reset Arrivals
        self.reset_arrivals()

        if self.analytics:
            self.most_recent_trade = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.spoof_orders = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.sell_orders = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.best_buys = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.best_asks = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.sell_above_best = []
            self.spoofer_quantity = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.spoofer_value = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.trade_volume = {key: 0 for key in range(0, self.sim_time + 1)}
            self.mid_prices = {key: np.nan for key in range(0, self.sim_time + 1)}

        end = self.run_until_next_SP_arrival()
        if end:
            input()
        #     raise ValueError("An episode without spoofer. Length of an episode should be set large.")

        return 0, {}

    def reset_arrivals(self):
        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = self.sim_time
        # self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
        self.arrival_times = self.sampled_arr
        self.arrival_index = 0

        self.arrivals_SP = defaultdict(list)
        if self.spoofer_arrivals != None:
            self.arrival_times_SP = self.spoofer_arrivals
        else:
            self.arrival_times_SP = sample_arrivals(self.lamSP, self.sim_time)

        self.arrival_index_SP = 0

        for agent_id in range(self.num_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

        #Xintong paper - Spoofer arrives after timestep 1000
        self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1000].append(self.num_agents)
        self.arrival_index_SP += 1

    def step(self):
        if self.time < self.sim_time:
            # Only matters for first iteration through.
            if len(self.arrivals_SP[self.time]) != 0:
                # input(random.random())
                self.SP_step()
            else:
                self.agents_step()
            
            reward = self.market_step(agent_only=False)
            self.time += 1
            end = self.run_until_next_SP_arrival()
            if end:
                return self.end_sim()
            return reward
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
                    random.seed(self.time + self.random_seed[self.time])
                    side = random.choice([BUY, SELL])
                    orders = agent.take_action(side, self.random_seed[self.time])
                    market.add_orders(orders)
                   
                    if self.arrival_index == self.arrivals_sampled:
                        self.arrival_times = self.sampled_arr
                        self.arrival_index = 0
                    self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + self.time].append(agent_id)
                    self.arrival_index += 1
                
        else:
            self.end_sim()

    def SP_step(self):
        for market in self.markets:
            market.event_queue.set_time(self.time)
            market.withdraw_all(self.num_agents)
            random.seed(self.time + self.random_seed[self.time])
            side = random.choice([BUY, SELL])
            order = self.spoofer.take_action(side, self.random_seed[self.time])

            market.add_orders(order)
            
            if self.arrival_index_SP == self.arrivals_sampled:
                self.arrival_times_SP = self.spoofer_arrivals
                self.arrival_index_SP = 0
            self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1 + self.time].append(self.num_agents)
            self.arrival_index_SP += 1
    
    def market_step(self, agent_only=True):
        for market in self.markets:
            new_orders = market.step()
            for matched_order in new_orders:
                agent_id = matched_order.order.agent_id
                quantity = matched_order.order.order_type * matched_order.order.quantity
                cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
                if agent_id == self.num_agents:
                    self.spoofer.update_position(quantity, cash)
                else:
                    self.agents[agent_id].update_position(quantity, cash)

            # ANALYTICAL DATA
            if self.analytics:
                self.trade_volume[self.time] = len(new_orders) // 2
                if len(self.markets[0].matched_orders) > 0:
                    self.most_recent_trade[self.time] = self.markets[0].matched_orders[-1].price
                if not math.isinf(self.markets[0].order_book.sell_unmatched.peek()):
                    self.best_asks[self.time] = self.markets[0].order_book.sell_unmatched.peek()
                if not math.isinf(self.markets[0].order_book.buy_unmatched.peek()):
                    self.best_buys[self.time] = self.markets[0].order_book.buy_unmatched.peek() 
                if not math.isinf(self.markets[0].order_book.sell_unmatched.peek()) and not math.isinf(self.markets[0].order_book.buy_unmatched.peek()):
                    self.mid_prices[self.time] = (self.best_asks[self.time] + self.best_buys[self.time]) / 2
                if len(self.markets[0].matched_orders) > 0:
                    self.most_recent_trade[self.time] = self.markets[0].matched_orders[-1].price
            
                # SPOOFER ANALYTICS
                self.spoofer_quantity[self.time] = self.spoofer.position
            
            if not agent_only:
                estimated_fundamental = self.spoofer.estimate_fundamental()
                current_value = self.spoofer.position * estimated_fundamental + self.spoofer.cash
                

                return current_value

    def end_sim_summarize(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        values = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            values[agent_id] = agent.get_pos_value() + agent.position * fundamental_val + agent.cash

        values[self.num_agents] = self.spoofer.position * fundamental_val + self.spoofer.cash
        # print(f'At the end of the simulation we get {values}')

    def end_sim(self):       
        estimated_fundamental = self.spoofer.estimate_fundamental()
        current_value = self.spoofer.position * estimated_fundamental + self.spoofer.cash
        return current_value

    def run_until_next_SP_arrival(self):
        while len(self.arrivals_SP[self.time]) == 0 and self.time < self.sim_time:
            self.agents_step()
            self.market_step(agent_only=True)
            # print(self.markets[0].order_book.observe())
            self.time += 1

        if self.time >= self.sim_time:
            return True
        else:
            return False

    def run_agents_only(self):
        for t in range(int(0.1 * self.sim_time)):
            if self.arrivals[t]:
                self.agents_step()
                self.market_step(agent_only=True)
            self.time += 1

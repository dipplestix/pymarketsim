import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

import random
from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.mean_reverting import GaussianMeanReverting
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.spoofer import SpoofingAgent
import torch.distributions as dist
import torch
from collections import defaultdict

def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps

class SPEnv(gym.Env):
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_assets: int = 1,
                 lam: float = 0.1,
                 lamSP: float = 0.1,
                 mean: float = 100,
                 r: float = 0.05,
                 shock_var: float = 10,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 order_size=100, # the size of regular order: NEED TUNING
                 spoofing_size=100, # the size of spoofing order: NEED TUNING
                 normalizers = None # normalizer for obs: NEED TUNING
                 ):

        # MarketSim Setup
        if shade is None:
            shade = [10, 30]
        self.num_agents = num_background_agents
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.time = 0

        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
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
            fundamental = GaussianMeanReverting(mean=mean, final_time=sim_time+1, r=r, shock_var=shock_var)
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        # Set up for regular traders.
        self.agents = {}
        for agent_id in range(num_background_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

            self.agents[agent_id] = (
                ZIAgent(
                    agent_id=agent_id,
                    market=self.markets[0],
                    q_max=q_max,
                    shade=shade,
                    pv_var=pv_var
                ))

        # Set up for spoofer.
        self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item()].append(self.num_agents)
        self.arrival_index_SP += 1
        self.spoofer = SpoofingAgent(
            agent_id=self.num_agents,
            market=self.markets[0],
            q_max=q_max,
            pv_var=pv_var,
            order_size=order_size,
            spoofing_size=spoofing_size,
            normalizers=normalizers
        )

        # Gym Setup
        """
        Given a market state s when the self agent arrives at time t, 
        the agent receives an observation O(s) that includes the number of time steps left T − t, 
        the current fundamental value rt, 
        the current best BID and ASK price in the limit order book (if any), 
        the self agent’s inventory I, 
        the self agent’s cash,
        """
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32) # price for regular order and price for spoofing

    def get_obs(self):
        return self.observation

    def update_obs(self): # TODO: Check if this observation works.
        self.time_left = self.sim_time - self.time
        self.fundamental_value = self.markets[0].fundamental.get_value_at(self.time)
        self.best_ask = self.markets[0].order_book.get_best_ask()
        self.best_bid = self.markets[0].order_book.get_best_bid()
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

        if self.normalizers is None:
            print("No normalizer warning!")
            return np.array([time_left, fundamental_value, best_ask, best_bid, SPinvt, SPcash])

        time_left /= self.sim_time
        fundamental_value /= self.normalizers["fundamental"]
        if math.isinf(best_ask):
            best_ask = 1
        else:
            best_ask /= self.normalizers["fundamental"]

        if math.isinf(best_bid):
            best_bid = 0
        else:
            best_bid /= self.normalizers["fundamental"]

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

        # Run until the spoofer enters.
        _ = self.run_until_next_SP_arrival()
        self.run_agents_only()

        return self.get_obs(), {}

    def reset_arrivals(self):
        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.arrivals_SP = defaultdict(list)
        self.arrival_times_SP = sample_arrivals(self.lamSP, self.arrivals_sampled)
        self.arrival_index_SP = 0

        for agent_id in range(self.num_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

        self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item()].append(self.num_agents)
        self.arrival_index_SP += 1


    def step(self, action):
        if self.time < self.sim_time:
            reward = self.SP_step(action)
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
        if len(agents) != 0:
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
                for matched_order in new_orders:
                    agent_id = matched_order.order.agent_id
                    quantity = matched_order.order.order_type * matched_order.order.quantity
                    cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
                    if agent_id == self.num_agents:
                        self.spoofer.update_position(quantity, cash)
                    else:
                        self.agents[agent_id].update_position(quantity, cash)

    def SP_step(self, action):
        for market in self.markets:
            market.event_queue.set_time(self.time)
            market.withdraw_all(self.num_agents)
            orders = self.spoofer.take_action(action)
            market.add_orders(orders)

            if self.arrival_index_SP == self.arrivals_sampled:
                self.arrival_times_SP = sample_arrivals(self.lamSP, self.arrivals_sampled)
                self.arrival_index_SP = 0
            self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1 + self.time].append(self.num_agents)
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
            reward = current_value - self.spoofer.last_value
            self.spoofer.last_value = reward #TODO: Check if we need to normalize the reward

        return reward


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
        reward = current_value - self.spoofer.last_value
        return self.get_obs(), reward, True, False, {}


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



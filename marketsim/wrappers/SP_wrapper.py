import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

import random
from fourheap.constants import BUY, SELL
from market.market import Market
from fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from agent.zero_intelligence_agent import ZIAgent
from agent.spoofer import SpoofingAgent
from wrappers.metrics import volume_imbalance, queue_imbalance, signed_volume, realized_volatility, relative_strength_index, midprice_move
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
        self.mean = mean
        self.shock_var = shock_var
        self.r = r
        self.markets = []
        for _ in range(num_assets):
            fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time+1, r=r, shock_var=shock_var)
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
        1.the agent receives an observation O(s) that includes the number of time steps left T − t, 
        2.the current fundamental value rt, 
        3.the current best BID price in the limit order book (if any), 
        4.the current best ASK price in the limit order book (if any), 
        5.the self agent’s inventory I, 
        ------
        Extra from Rahul's paper:
        6.Mid-price move,
        7.Volume imbalance
        8.Queue imbalance,
        9.Volatility,
        10.Relative strength index,
        ------
        11. Private values 2 * q_max
        """
        lower_bound = np.zeros(10 + 2 * q_max)
        for i in [4,5,6,7]:
            lower_bound[i] = -1
        uppper_bound = np.ones(10 + 2 * q_max)
        self.observation_space = spaces.Box(low=lower_bound,
                                            high=uppper_bound,
                                            shape=(10 + 2*q_max,),
                                            dtype=np.float64) # Need rescale the obs.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32) # price for regular order and price for spoofing

    def get_obs(self):
        return self.observation

    def update_obs(self):
        time_left = self.sim_time - self.time
        fundamental_value = self.markets[0].fundamental.get_value_at(self.time)
        best_ask = self.markets[0].order_book.get_best_ask()
        best_bid = self.markets[0].order_book.get_best_bid()
        SPinvt = self.spoofer.position

        midprice_delta = midprice_move(self.markets[0])
        vol_imbalance = volume_imbalance(self.markets[0])
        que_imbalance = queue_imbalance(self.markets[0])
        vr = realized_volatility(self.markets[0])
        rsi = relative_strength_index(self.markets[0])

        pv = self.spoofer.pv.values.numpy()

        self.observation = self.normalization(
            time_left=time_left,
            fundamental_value=fundamental_value,
            best_ask=best_ask,
            best_bid=best_bid,
            SPinvt=SPinvt,
            midprice_delta=midprice_delta,
            vol_imbalance=vol_imbalance,
            que_imbalance=que_imbalance,
            vr=vr,
            rsi=rsi,
            pv=pv)

    def normalization(self,
                      time_left: int,
                      fundamental_value: float,
                      best_ask: float,
                      best_bid: float,
                      SPinvt: float,
                      midprice_delta: float,
                      vol_imbalance: float,
                      que_imbalance: float,
                      vr: float,
                      rsi: float,
                      pv:np.ndarray):

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

        # -------
        midprice_delta /= 1e2  # TODO: need to tune
        rsi /= 100
        pv /= 5e5

        obs1 = np.array([time_left,
                         fundamental_value,
                         best_ask,
                         best_bid,
                         SPinvt,
                         midprice_delta,
                         vol_imbalance * 10,
                         que_imbalance * 10,
                         vr,
                         rsi])

        obs = np.concatenate((obs1, pv))

        return obs

    def reset(self, seed=None, options=None):
        self.time = 0
        self.observation = None

        # Reset the markets
        for market in self.markets:
            fundamental = LazyGaussianMeanReverting(mean=self.mean, final_time=self.sim_time + 1, r=self.r, shock_var=self.shock_var)
            market.reset(fundamental=fundamental)

        # Reset the agents
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.reset()

        # Reset spoofer
        self.spoofer.reset()

        # Reset Arrivals
        self.reset_arrivals()

        # Run until the spoofer enters.
        self.run_agents_only()
        end = self.run_until_next_SP_arrival()

        if end:
            raise ValueError("An episode without spoofer. Length of an episode should be set large.")


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
            self.SP_step(action)
            self.agents_step()
            reward = self.market_step(agent_only=False)
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

            if not agent_only:
                fundamental_val = self.markets[0].get_final_fundamental()
                current_value = self.spoofer.position * fundamental_val + self.spoofer.cash + self.spoofer.get_pos_value()
                reward = current_value - self.spoofer.last_value
                self.spoofer.last_value = current_value  # TODO: Check if we need to normalize the reward

                return reward / self.normalizers["fundamental"]  # TODO: Check if this normalizer works. Anri: 1e2

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
            self.market_step(agent_only=True)
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
                self.market_step(agent_only=True)
            self.time += 1



import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

import random
from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.agent.noise_ZI_agent import ZIAgent
from marketsim.agent.informed_ZI import ZIAgent as InformedZIAgent
from marketsim.agent.market_maker_beta import MMAgent
from marketsim.wrappers.metrics import volume_imbalance, queue_imbalance, realized_volatility, relative_strength_index, midprice_move
import torch.distributions as dist
import torch
from collections import defaultdict

def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 10000 sampled time steps

class MMEnv(gym.Env):
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_assets: int = 1,
                 lam: float = 75e-3,
                 lamMM: float = 5e-3,
                 informedZI = False,
                 mean: float = 1e5,
                 r: float = 0.05,
                 shock_var: float = 5e6,
                 q_max: int = 10,
                 est_var: float = 1e6,
                 pv_var: float = 5e6,
                 shade=None,
                 n_levels: int=21,
                 total_volume: int=100,
                 xi: float = 50, # rung size
                 omega: float = 10, #spread
                 beta_params: dict = None,
                 policy=False,
                 normalizers=None
                 ):

        # MarketSim Setup
        if shade is None:
            shade = [10, 30]
        self.num_agents = num_background_agents
        self.total_num_agents = self.num_agents + 1
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.time = 0

        # Regular traders
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
        self.arrival_index = 0

        # MM
        self.lamMM = lamMM
        self.arrivals_MM = defaultdict(list)
        self.arrival_times_MM  = sample_arrivals(lamMM, self.arrivals_sampled)
        self.arrival_index_MM = 0
        self.normalizers = normalizers

        self.mean = mean
        self.shock_var = shock_var
        self.r = r
        self.markets = []
        if num_assets > 1:
            raise NotImplemented("Only support single market currently")

        for _ in range(num_assets):
            fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time+1, r=r, shock_var=shock_var)
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        # Set up for regular traders.
        self.agents = {}
        for agent_id in range(num_background_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

            if informedZI and agent_id >= int(num_background_agents / 2):
                self.agents[agent_id] = (
                    InformedZIAgent(
                        agent_id=agent_id,
                        market=self.markets[0],
                        q_max=q_max,
                        shade=shade,
                        pv_var=pv_var
                    ))
            else:
                self.agents[agent_id] = (
                    ZIAgent(
                        agent_id=agent_id,
                        market=self.markets[0],
                        q_max=q_max,
                        shade=shade,
                        pv_var=pv_var,
                        est_var=est_var
                    ))

        # Set up for market makers.
        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item()].append(self.num_agents)
        self.arrival_index_MM += 1
        self.MM = MMAgent(
                agent_id=self.num_agents,
                market=self.markets[0],
                n_levels=n_levels,
                total_volume=total_volume,
                xi=xi,
                omega=omega,
                beta_params=beta_params,
                policy=policy
            )

        # Metrics
        self.spreads = []
        self.midprices = []
        self.inventory = []
        self.value_MM = 0
        self.total_quantity = 0
        self.MM_quantity = 0

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
        """
        # self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0]),
        #                                     high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        #                                     shape=(10,),
        #                                     dtype=np.float64) # Need rescale the obs.

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, -1.0]),
                                            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                                            shape=(5,),
                                            dtype=np.float64)  # Need rescale the obs.

        # self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64) # a_buy, b_buy, a_sell, b_sell
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

    def get_obs(self):
        return self.observation

    def update_obs(self):
        time_left = self.sim_time - self.time
        fundamental_value = self.markets[0].fundamental.get_value_at(self.time)
        best_ask = self.markets[0].order_book.get_best_ask()
        best_bid = self.markets[0].order_book.get_best_bid()
        MMinvt = self.MM.position

        midprice_delta = midprice_move(self.MM.market)
        vol_imbalance = volume_imbalance(self.MM.market)
        que_imbalance = queue_imbalance(self.MM.market)
        vr = realized_volatility(self.MM.market)
        rsi = relative_strength_index(self.MM.market)

        self.observation = self.normalization(
            time_left=time_left,
            fundamental_value=fundamental_value,
            best_ask=best_ask,
            best_bid=best_bid,
            MMinvt=MMinvt,
            midprice_delta=midprice_delta,
            vol_imbalance=vol_imbalance,
            que_imbalance=que_imbalance,
            vr=vr,
            rsi=rsi)

    def normalization(self,
                      time_left: int,
                      fundamental_value: float,
                      best_ask: float,
                      best_bid: float,
                      MMinvt: float,
                      midprice_delta: float,
                      vol_imbalance: float,
                      que_imbalance: float,
                      vr: float,
                      rsi: float):

        if self.normalizers is None:
            print("No normalizer warning!")
            return np.array([time_left, fundamental_value, best_ask, best_bid, MMinvt])

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

        MMinvt /= self.normalizers["invt"]

        #-------
        midprice_delta /= 1e2 # TODO: need to tune
        rsi /= 100

        # return np.array([time_left,
        #                  fundamental_value,
        #                  best_ask,
        #                  best_bid,
        #                  MMinvt,
        #                  midprice_delta,
        #                  vol_imbalance * 10,
        #                  que_imbalance * 10,
        #                  vr,
        #                  rsi])

        return np.array([time_left,
                         fundamental_value,
                         best_ask,
                         best_bid,
                         MMinvt])


    def reset(self, seed=None, options=None):
        self.time = 0
        self.observation = None

        # Reset the markets
        for market in self.markets:
            fundamental = LazyGaussianMeanReverting(mean=self.mean,
                                                final_time=self.sim_time + 1,
                                                r=self.r,
                                                shock_var=self.shock_var)
            market.reset(fundamental=fundamental)

        # Reset the agents
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.reset()

        # Reset MM
        self.MM.reset()

        # Metrics
        self.spreads = []
        self.midprices = []
        self.inventory = []
        self.value_MM = 0
        self.total_quantity = 0
        self.MM_quantity = 0

        # Reset Arrivals
        self.reset_arrivals()

        # Run until the MM enters.
        # self.run_agents_only() #TODO: run agent only could exclude the arrival of MM.
        _, end = self.run_until_next_MM_arrival()

        if end:
            raise ValueError("An episode without MM. Length of an episode should be set large.")


        # print("OBS RET:", self.get_obs())
        return self.get_obs(), {}



    def reset_arrivals(self):
        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.arrivals_MM = defaultdict(list)
        self.arrival_times_MM = sample_arrivals(self.lamMM, self.arrivals_sampled)
        self.arrival_index_MM = 0

        for agent_id in range(self.num_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item()].append(self.num_agents)
        self.arrival_index_MM += 1


    def step(self, action):
        if self.time < self.sim_time:
            self.MM_step(action)
            self.agents_step()
            self.market_step(agent_only=False)
            self.time += 1
            reward, end = self.run_until_next_MM_arrival()
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

    def MM_step(self, action):
        for market in self.markets:
            market.event_queue.set_time(self.time)
            market.withdraw_all(self.num_agents)
            orders = self.MM.take_action(action)
            # print("MM orders:", len(orders), action)
            market.add_orders(orders)

            if self.arrival_index_MM == self.arrivals_sampled:
                self.arrival_times_MM = sample_arrivals(self.lamMM, self.arrivals_sampled)
                self.arrival_index_MM = 0
            self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item() + 1 + self.time].append(self.num_agents)
            self.arrival_index_MM += 1

    def market_step(self, agent_only=True, verbose=False):
        for market in self.markets:
            new_orders = market.step()
            for matched_order in new_orders:
                agent_id = matched_order.order.agent_id
                quantity = matched_order.order.order_type * matched_order.order.quantity
                cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
                if agent_id == self.num_agents:
                    self.MM.update_position(quantity, cash)
                else:
                    self.agents[agent_id].update_position(quantity, cash)

                # Record
                self.total_quantity += abs(quantity)
                if agent_id == self.num_agents:
                    self.MM_quantity += abs(quantity)

            # Record stats
            best_ask = market.order_book.get_best_ask()
            best_bid = market.order_book.get_best_bid()
            self.spreads.append(best_ask - best_bid)
            self.midprices.append((best_ask + best_bid) / 2)
            self.inventory.append(self.MM.position)

            if not agent_only:
                if verbose:
                    print("----midprice:", (best_ask + best_bid) / 2)
                    print("----fundamental:", self.MM.estimate_fundamental())
                    print("----final fundamental:", market.get_final_fundamental())
                    print("----matched orders:", new_orders)
                    print("----self.MM.last_value:", self.MM.last_value)
                    print("----Position:", self.MM.position)
                    print("----Cash:", self.MM.cash)
                    print("----Best ask：", self.MM.market.order_book.get_best_ask())
                    print("----Best bid：", self.MM.market.order_book.get_best_bid())
                    print("----Bids：", self.MM.market.order_book.buy_unmatched)
                    print("----Asks：", self.MM.market.order_book.sell_unmatched)


    def end_sim_summarize(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        values = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            values[agent_id] = agent.get_pos_value() + agent.position * fundamental_val + agent.cash

        values[self.num_agents] = self.MM.position * fundamental_val + self.MM.cash

    def end_sim(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        current_value = self.MM.position * fundamental_val + self.MM.cash
        reward = current_value - self.MM.last_value
        self.MM.last_value = current_value
        self.value_MM = current_value

        return self.get_obs(), reward / self.normalizers["reward"], True, False, {}


    def run_until_next_MM_arrival(self):
        while len(self.arrivals_MM[self.time]) == 0 and self.time < self.sim_time:
            self.agents_step()
            self.market_step(agent_only=True)
            self.time += 1

        if self.time >= self.sim_time:
            return 0, True
        else:
            fundamental_val = self.markets[0].get_final_fundamental()
            current_value = self.MM.position * fundamental_val + self.MM.cash
            reward = current_value - self.MM.last_value
            self.MM.last_value = current_value
            self.update_obs()

            return reward / self.normalizers["reward"], False


    def run_agents_only(self):
        for t in range(int(0.01 * self.sim_time)):
            if self.arrivals[t]:
                self.agents_step()
                self.market_step(agent_only=True)
            self.time += 1

    def get_stats(self):
        stats = {}
        stats["spreads"] = self.spreads.copy()
        stats["midprices"] = self.midprices.copy()
        stats["inventory"] = self.inventory.copy()
        stats["total_quantity"] = self.total_quantity
        stats["MM_quantity"] = self.MM_quantity
        stats["MM_value"] = self.value_MM

        return stats

    def compute_social_welfare(self):
        values = []
        fundamental_val = self.markets[0].get_final_fundamental()
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            values.append(agent.get_pos_value() + agent.position * fundamental_val + agent.cash)

        values.append(self.MM.position * fundamental_val + self.MM.cash)

        return sum(values)


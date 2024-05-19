import random
import numpy as np
from fourheap.constants import BUY, SELL
from market.market import Market
from fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from agent.zero_intelligence_agent import ZIAgent
from agent.hbl_agent import HBLAgent
import torch.distributions as dist
import torch
from collections import defaultdict

def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps


class SimulatorSampledArrival:
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_assets: int = 1,
                 lam: float = 5e-3,
                 mean: float = 100,
                 r: float = .05,
                 shock_var: float = 10,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 eta: float = 0.2,
                 hbl_agent: bool = False,
                 pvalues = None,
                 sampled_arr = None,
                 fundamental = None,
                 ):

        if shade is None:
            shade = [10, 30]
        self.num_agents = num_background_agents
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.time = 0
        self.hbl_agent = hbl_agent
        self.most_recent_trade = {key: np.nan for key in range(0, self.sim_time + 1)}
        

        self.b = None

        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        if sampled_arr != None:
            self.arrival_times = sampled_arr
        else:
            self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.markets = []
        for _ in range(num_assets):
            if fundamental != None:
                fundamental = fundamental
            else:
                fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        self.agents = {}
        #TEMP FOR HBL TESTING
        if not self.hbl_agent:
            for agent_id in range(num_background_agents + 1):
                self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
                self.arrival_index += 1
                self.agents[agent_id] = (
                    ZIAgent(
                        agent_id=agent_id,
                        market=self.markets[0],
                        q_max=q_max,
                        shade=shade,
                        pv_var=pv_var,
                    ))
        else:
            for agent_id in range(12):
                if pvalues != None:
                    pv = pvalues[agent_id]
                else:
                    pv = None
                self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
                self.arrival_index += 1
                self.agents[agent_id] = (
                    ZIAgent(
                        agent_id=agent_id,
                        market=self.markets[0],
                        q_max=q_max,
                        shade=shade,
                        pv_var=pv_var,
                        pv=pv
                    ))
            for agent_id in range(12,25):
                if pvalues != None:
                    pv = pvalues[agent_id]
                else:
                    pv = None
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
                    pv=pv
                ))

    def step(self):
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
                for matched_order in new_orders:
                    agent_id = matched_order.order.agent_id
                    quantity = matched_order.order.order_type*matched_order.order.quantity
                    cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                    self.agents[agent_id].update_position(quantity, cash)

        else:
            self.end_sim()

    def end_sim(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        values = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            values[agent_id] = agent.get_pos_value() + agent.position*fundamental_val + agent.cash
        # print(f'At the end of the simulation we get {values}')
        return values

    def run(self):
        for t in range(self.sim_time):
            if self.arrivals[t]:
                # print(t, self.arrivals[t])
                # input()
                try:
                    # print(f'It is time {t}')
                    self.step()
                    # print(self.markets[0].order_book.observe())
                    # print("----Best ask：", self.markets[0].order_book.get_best_ask())
                    # print("----Best bid：", self.markets[0].order_book.get_best_bid())
                    # print("----Bids：", self.markets[0].order_book.buy_unmatched)
                    # print("----Asks：", self.markets[0].order_book.sell_unmatched)
                except KeyError:
                    print(self.arrivals[self.time])
                    return self.markets
            if len(self.markets[0].matched_orders) > 0:
                if self.time > 9900 and self.markets[0].matched_orders[-1].price != self.most_recent_trade[self.time - 1]:
                    a = self.markets[0].matched_orders[-1].price
                    self.b = self.agents[0].estimate_fundamental()
                self.most_recent_trade[self.time] = self.markets[0].matched_orders[-1].price

            self.time += 1
        self.step()

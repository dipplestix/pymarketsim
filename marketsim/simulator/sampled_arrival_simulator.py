import random
import math
from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.agent.zero_intelligence_agent import ZIAgent
import torch.distributions as dist
import torch
from collections import defaultdict


class SimulatorSampledArrival:
    def __init__(self, num_agents: int, sim_time: int, num_assets: int = 1, lam=0.1, mean=100, r=.6, shock_var=10, agents=None, markets=None):
        self.num_agents = num_agents
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.time = 0

        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.markets = markets
        self.agents = agents

        #Liquidity tests
        self.price_spread = {"buy": set(), "sell": set()}
        self.ob_summary = {"buy": [], "sell": []}
        self.ob_max_spread = []
        self.fundamental_evol = []
        self.spread = []

        if markets == None:
            self.markets = []
            for _ in range(num_assets):
                fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
                self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        if agents == None:
            self.agents = {}
            for agent_id in range(num_agents):
                self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
                self.arrival_index += 1

                self.agents[agent_id] = (
                    ZIAgent(
                        agent_id=agent_id,
                        market=self.markets[0],
                        q_max=20,
                        offset=12,
                        shade=[10, 30]
                    ))
        else:
            for agent in self.agents:
                self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent)
                self.arrival_index += 1

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
                
                self.price_spread = {"buy": set(), "sell": set()}
                for price, order_id in (market.order_book.buy_unmatched.heap):
                    self.price_spread["buy"].add(price)
                for price, order_id in (market.order_book.sell_unmatched.heap):
                    self.price_spread["sell"].add(price)
                
                self.ob_summary["buy"].append(len(self.price_spread["buy"]))
                self.ob_summary["sell"].append(len(self.price_spread["sell"]))
                
                if not self.price_spread["buy"]:
                    buy_spread = 0
                else:
                    buy_spread = max(self.price_spread["buy"]) - min(self.price_spread["buy"])
               
                if not self.price_spread["sell"]:
                    sell_spread = 0
                else:
                    sell_spread = max(self.price_spread["sell"]) - min(self.price_spread["sell"])
                
                self.spread.append(abs(market.order_book.buy_unmatched.peek() - market.order_book.sell_unmatched.peek()))
                self.ob_max_spread.append((buy_spread, sell_spread))
                self.fundamental_evol.append(market.get_fundamental_value())
        else:
            self.end_sim()

    def end_sim(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        values = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            values[agent_id] = agent.get_pos_value() + agent.position*fundamental_val + agent.cash
        # print(f'At the end of the simulation we get {values}')

    def run(self):
        counter = 0
        for t in range(self.sim_time):
            if self.arrivals[t]:
                try:
                    # print(f'It is time {t}')
                    self.step()
                except KeyError:
                    print(self.arrivals[self.time])
                    return self.markets
                counter += 1
            self.time += 1
        self.step()


def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps

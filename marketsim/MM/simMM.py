import random
from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.market_maker import MMAgent
from marketsim.agent.market_maker_beta import MMAgent as MMbetaAgent
import torch.distributions as dist
import torch
from collections import defaultdict


class SimulatorSampledArrival_MM:
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int = 12000,
                 num_assets: int = 1,
                 lam: float = 1e-3,
                 lamMM: float = 5,
                 mean: float = 1e5,
                 r: float = 0.05,
                 shock_var: float = 5e6,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 xi: float = 100,  # rung size
                 omega: float = 64,  # spread,
                 n_levels: int = 101,
                 total_volume: int = 100,
                 K: int = 100, # n_level - 1
                 beta_params: dict = None,
                 policy=None,
                 beta_MM=False,
                 inv_driven=False,
                 w0=5,
                 p=2,
                 k_min=5,
                 k_max=20,
                 max_position=100
                 ):

        if shade is None:
            shade = [10, 30]
        self.num_agents = num_background_agents + 1
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
        self.arrival_times_MM = sample_arrivals(lamMM, self.arrivals_sampled)
        self.arrival_index_MM = 0
        self.warm_up_time = 0.01 * self.sim_time


        self.markets = []
        for _ in range(num_assets):
            fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

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

        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item()].append(self.num_agents)
        self.arrival_index_MM += 1

        if beta_MM:
            self.MM = MMbetaAgent(
                agent_id=self.num_agents,
                market=self.markets[0],
                n_levels=n_levels,
                total_volume=total_volume,
                xi=xi,
                omega=omega,
                beta_params=beta_params,
                policy=policy,
                inv_driven=inv_driven,
                w0=w0,
                p=p,
                k_min=k_min,
                k_max=k_max,
                max_position=max_position
            )
        else:
            self.MM = MMAgent(
                agent_id=self.num_agents,
                market=self.markets[0],
                K=K,
                xi=xi,
                omega=omega
            )
        self.agents[self.num_agents] = self.MM


    def step(self):
        agents = self.arrivals[self.time] + self.arrivals_MM[self.time]
        if self.time < self.sim_time:
            for market in self.markets:
                market.event_queue.set_time(self.time)
                for agent_id in agents:
                    agent = self.agents[agent_id]
                    market.withdraw_all(agent_id)
                    if agent_id == self.num_agents: # MM
                        if self.time < self.warm_up_time: # Warm up the simulator
                            continue
                        orders = agent.take_action()
                        market.add_orders(orders)
                        if self.arrival_index_MM == self.arrivals_sampled:
                            self.arrival_times_MM = sample_arrivals(self.lamMM, self.arrivals_sampled)
                            self.arrival_index_MM = 0
                        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item() + 1 + self.time].append(
                            self.num_agents)
                        self.arrival_index_MM += 1
                    else: # Regular agents.
                        side = random.choice([BUY, SELL])
                        orders = agent.take_action(side)
                        market.add_orders(orders)

                        if self.arrival_index == self.arrivals_sampled:
                            self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
                            self.arrival_index = 0
                        self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + self.time].append(agent_id)
                        self.arrival_index += 1
                    print("time:", self.time, "orders:", orders)

                new_orders = market.step()
                print("time:", self.time, "orders:", new_orders)
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
            if agent_id == self.num_agents:
                values[agent_id] = agent.position * fundamental_val + agent.cash
            else:
                values[agent_id] = agent.get_pos_value() + agent.position*fundamental_val + agent.cash
        # print(f'At the end of the simulation we get {values}')

    def run(self):
        counter = 0
        print("Arrival Time:", self.arrivals)
        print("Arrival Time MM:", self.arrivals_MM)
        for t in range(self.sim_time):
            if self.arrivals[t]:
                try:
                    print(f'------------It is time {t}-------------')
                    self.step()
                    print("MM position:", self.MM.position)
                    print("MM cash:", self.MM.cash)
                    # print(self.markets[0].order_book.observe())
                    print("----Best ask：", self.markets[0].order_book.get_best_ask())
                    print("----Best bid：", self.markets[0].order_book.get_best_bid())
                    print("----Bids：", self.markets[0].order_book.buy_unmatched)
                    print("----Asks：", self.markets[0].order_book.sell_unmatched)
                except KeyError:
                    print(self.arrivals[self.time])
                    return self.markets
                counter += 1
            self.time += 1
        self.step()


def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps


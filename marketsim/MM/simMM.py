import random
from fourheap.constants import BUY, SELL
from market.market import Market
from fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from agent.zero_intelligence_agent import ZIAgent
from agent.market_maker import MMAgent
from agent.market_maker_beta import MMAgent as MMbetaAgent
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
                 inv_driven=False, # enable inventory driven policy
                 w0=5,
                 p=2,
                 k_min=5,
                 k_max=20,
                 max_position=100
                 ):
        
        if random_seed != 0:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            # np.random.seed(random_seed)

        if shade is None:
            shade = [250, 500]
        self.num_background_agents = num_background_agents
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

        self.mean = mean
        self.shock_var = shock_var
        self.r = r
        self.markets = []
        if num_assets > 1:
            raise NotImplemented("Only support single market currently")


        self.markets = []
        for _ in range(num_assets):
            fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time+1, r=r, shock_var=shock_var)
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
                    pv_var=pv_var,
                    random_seed=random.randint(1,4096)
                ))

        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item()].append(self.num_background_agents)
        self.arrival_index_MM += 1

        if beta_MM: # beta policy
            self.MM = MMbetaAgent(
                agent_id=self.num_background_agents,
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
        else: # ladder policy
            self.MM = MMAgent(
                agent_id=self.num_background_agents,
                market=self.markets[0],
                K=K,
                xi=xi,
                omega=omega,
                random_seed=random.randint(1,4096)
            )
        self.agents[self.num_background_agents] = self.MM

        # Metrics
        self.spreads = []
        self.midprices = []
        self.inventory = []
        self.value_MM = 0
        self.total_quantity = 0
        self.MM_quantity = 0

    def step(self, agent_only=False):
        agents = self.arrivals[self.time]
        if not agent_only:
            agents += self.arrivals_MM[self.time]

        if self.time < self.sim_time:
            for market in self.markets:
                market.event_queue.set_time(self.time)
                for agent_id in agents:
                    agent = self.agents[agent_id]
                    market.withdraw_all(agent_id)
                    if agent_id == self.num_background_agents: # MM
                        orders = agent.take_action()
                        # print("MM Orders:", orders)
                        market.add_orders(orders)
                        if self.arrival_index_MM == self.arrivals_sampled:
                            self.arrival_times_MM = sample_arrivals(self.lamMM, self.arrivals_sampled)
                            self.arrival_index_MM = 0
                        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item() + 1 + self.time].append(agent_id)
                        self.arrival_index_MM += 1
                        # print(self.arrival_times_MM[self.arrival_index_MM].item() + 1 + self.time)
                    else: # Regular agents.
                        side = random.choice([BUY, SELL])
                        orders = agent.take_action(side)
                        market.add_orders(orders)

                        if self.arrival_index == self.arrivals_sampled:
                            self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
                            self.arrival_index = 0
                        self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + self.time].append(agent_id)
                        self.arrival_index += 1
                    # print("time:", self.time, "orders:", orders)

                new_orders = market.step()
                # print("time:", self.time, "orders:", new_orders)
                for matched_order in new_orders:
                    agent_id = matched_order.order.agent_id
                    quantity = matched_order.order.order_type*matched_order.order.quantity
                    cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                    self.agents[agent_id].update_position(quantity, cash)
                    # Record
                    self.total_quantity += abs(quantity)
                    if agent_id == self.num_background_agents:
                        self.MM_quantity += abs(quantity)

                # Record stats
                best_ask = market.order_book.get_best_ask()
                best_bid = market.order_book.get_best_bid()
                self.spreads.append(best_ask - best_bid)
                self.midprices.append((best_ask + best_bid)/2)
                self.inventory.append(self.MM.position)

        # else:
        #     self.end_sim()

    def end_sim(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        agent = self.agents[self.num_background_agents]
        self.value_MM = agent.position * fundamental_val + agent.cash
        # for agent_id in self.agents:
        #     agent = self.agents[agent_id]
        #     if agent_id == self.num_background_agents: # MM: does not have private values.
        #         self.values.append(agent.position * fundamental_val + agent.cash)
        #     else:
        #         self.values.append(agent.get_pos_value() + agent.position * fundamental_val + agent.cash)



    def run(self):
        counter = 0
        # print("Arrival Time:", self.arrivals)
        # print("Arrival Time MM:", self.arrivals_MM)
        for t in range(self.sim_time):
            if self.arrivals[t] + self.arrivals_MM[t]:
                # print(f'------------It is time {t}-------------')
                self.step()
                # print("MM position:", self.MM.position)
                # print("MM cash:", self.MM.cash)
                # print(self.markets[0].order_book.observe())
                # print("----Best ask：", self.markets[0].order_book.get_best_ask())
                # print("----Best bid：", self.markets[0].order_book.get_best_bid())
                # print("----Bids：", self.markets[0].order_book.buy_unmatched)
                # print("----Asks：", self.markets[0].order_book.sell_unmatched)
                counter += 1
            else:
                # Record stats
                best_ask = self.markets[0].order_book.get_best_ask()
                best_bid = self.markets[0].order_book.get_best_bid()
                self.spreads.append(best_ask - best_bid)
                self.midprices.append((best_ask + best_bid) / 2)
                self.inventory.append(self.MM.position)
            self.time += 1
        self.end_sim()
        #TODO: We can track the value of MM by using the true fundamental.
        stats = self.get_stats()
        return stats


    def reset(self, seed=None, options=None):
        self.time = 0

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

        # Reset Metrics
        self.spreads = []
        self.midprices = []
        self.inventory = []
        self.value_MM = 0
        self.total_quantity = 0
        self.MM_quantity = 0

        # Reset Arrivals
        self.reset_arrivals()

        # Run until the MM enters.
        # self.run_agents_only()



    def reset_arrivals(self):
        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.arrivals_MM = defaultdict(list)
        self.arrival_times_MM = sample_arrivals(self.lamMM, self.arrivals_sampled)
        self.arrival_index_MM = 0

        for agent_id in range(self.num_background_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

        # self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item()].append(self.num_background_agents)
        # self.arrival_index_MM += 1
        self.arrivals_MM[0].append(self.num_background_agents)


    def run_agents_only(self, all_time_steps=False):
        if all_time_steps:
            sim_time = self.sim_time
        else:
            sim_time = self.warm_up_time
        for t in range(int(sim_time)):
            if self.arrivals[t]:
                self.step(agent_only=True)
            self.time += 1

        if all_time_steps:
            return self.get_stats()


    def get_stats(self):
        stats = {}
        stats["spreads"] = self.spreads.copy()
        stats["midprices"] = self.midprices.copy()
        stats["inventory"] = self.inventory.copy()
        stats["total_quantity"] = self.total_quantity
        stats["MM_quantity"] = self.MM_quantity
        stats["MM_value"] = self.value_MM

        return stats




def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps


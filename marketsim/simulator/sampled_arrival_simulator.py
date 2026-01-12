import random
from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.hbl_agent import HBLAgent
import numpy as np
from collections import defaultdict


def sample_arrivals_numpy(p, num_samples):
    """Sample arrival times using numpy geometric distribution (faster than torch).

    Note: np.random.geometric is 1-based (number of trials), while torch.Geometric
    is 0-based (number of failures). We subtract 1 to match torch semantics.
    """
    return np.random.geometric(p, size=num_samples) - 1


class SimulatorSampledArrival:
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_assets: int = 1,
                 lam: float = 0.1,
                 mean: float = 100,
                 r: float = .05,
                 shock_var: float = 10,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 eta: float = 0.2,
                 hbl_agent: bool = False,
                 lam_r: float = None
                 ):

        if shade is None:
            shade = [10, 30]
        if lam_r is None:
            lam_r = lam

        self.num_agents = num_background_agents
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.lam_r = lam_r
        self.time = 0
        self.hbl_agent = hbl_agent
        self.r = r
        self.mean = mean

        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.initial_arrivals = sample_arrivals_numpy(lam, self.num_agents)
        self.arrival_times = sample_arrivals_numpy(lam_r, self.arrivals_sampled)
        self.arrival_index = 0

        # Precompute rho table for estimate_fundamental (performance optimization)
        self._rho_table = np.power(1 - r, np.arange(sim_time + 1, dtype=np.float64)[::-1])

        self.markets = []
        for _ in range(num_assets):
            fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        self.agents = {}
        # TEMP FOR HBL TESTING
        if not self.hbl_agent:
            for agent_id in range(num_background_agents + 1):
                self.arrivals[int(self.arrival_times[self.arrival_index])].append(agent_id)
                self.arrival_index += 1
                self.agents[agent_id] = (
                    ZIAgent(
                        agent_id=agent_id,
                        market=self.markets[0],
                        q_max=q_max,
                        shade=shade,
                        pv_var=pv_var,
                        eta=eta
                    ))
        #  expanded_zi
        # else:
        #     for agent_id in range(24):
        #         self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
        #         self.arrival_index += 1
        #         self.agents[agent_id] = (
        #             ZIAgent(
        #                 agent_id=agent_id,
        #                 market=self.markets[0],
        #                 q_max=q_max,
        #                 shade=shade,
        #                 pv_var=pv_var,
        #                 eta=eta
        #             ))
        #     for agent_id in range(24,25):
        #         self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
        #         self.arrival_index += 1
        #         self.agents[agent_id] = (HBLAgent(
        #             agent_id = agent_id,
        #             market = self.markets[0],
        #             pv_var = pv_var,
        #             q_max= q_max,
        #             shade = shade,
        #             L = 4,
        #             arrival_rate = self.lam
        #         ))

    def _get_cached_estimate(self):
        """Compute fundamental estimate once per timestep using precomputed rho table."""
        t = self.time
        rho = self._rho_table[t]
        val = self.markets[0].get_fundamental_value()
        return (1 - rho) * self.mean + rho * val

    def step(self):
        agents = self.arrivals[self.time]
        if self.time < self.sim_time:
            for market in self.markets:
                market.event_queue.set_time(self.time)
                # Cache the fundamental estimate after set_time (uses current time for fundamental)
                cached_estimate = self._get_cached_estimate()
                for agent_id in agents:
                    agent = self.agents[agent_id]
                    market.withdraw_all(agent_id)
                    orders = agent.take_action(estimate=cached_estimate)
                    market.add_orders(orders)
                    if self.arrival_index == self.arrivals_sampled:
                        self.arrival_times = sample_arrivals_numpy(self.lam_r, self.arrivals_sampled)
                        self.arrival_index = 0
                    self.arrivals[int(self.arrival_times[self.arrival_index]) + 1 + self.time].append(agent_id)
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
        counter = 0
        for t in range(self.sim_time):
            if self.arrivals[t]:
                try:
                    self.step()
                except KeyError:
                    print(self.arrivals[self.time])
                    return self.markets
                counter += 1
            self.time += 1
        self.step()


# Legacy function kept for compatibility with other modules
def sample_arrivals(p, num_samples):
    """Sample arrival times using numpy (legacy wrapper)."""
    return sample_arrivals_numpy(p, num_samples)

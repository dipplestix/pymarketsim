import random
from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.mean_reverting import GaussianMeanReverting
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.agent.zero_intelligence_agent import ZIAgent



class Simulator:
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_assets: int = 1,
                 lam: float = 0.1,
                 mean: float = 100,
                 r: float = .6,
                 shock_var=10,
                 q_max: int = 10,
                 zi_shade: List = [10, 30]):
        self.num_agents = num_background_agents
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.time = 0

        self.markets = []
        for _ in range(num_assets):
            fundamental = GaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
            # fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        self.agents = {}
        for agent_id in range(num_background_agents):
            self.agents[agent_id] = (
                ZIAgent(
                    agent_id=agent_id,
                    market=self.markets[0],
                    q_max=q_max,
                    shade=[10, 30]
                ))

    def step(self):
        # print(f'It is time step {self.time}')
        if self.time < self.sim_time:
            for market in self.markets:
                for agent_id in self.agents:
                    if random.random() <= self.lam:
                        agent = self.agents[agent_id]
                        market.withdraw_all(agent_id)
                        side = random.choice([BUY, SELL])
                        orders = agent.take_action(side)
                        # print(f'Agent {agent.agent_id} is entering the market and makes order {order}')
                        market.add_orders(orders)
                new_orders = market.step()
                for matched_order in new_orders:
                    agent_id = matched_order.order.agent_id
                    quantity = matched_order.order.order_type * matched_order.order.quantity
                    cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
                    self.agents[agent_id].update_position(quantity, cash)
            self.time += 1
        else:
            self.end_sim()

    def end_sim(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        values = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            values[agent_id] = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        # print(f'At the end of the simulation we get {values}')

    def run(self):
        for t in range(self.sim_time + 1):
            self.step()

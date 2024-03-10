import numpy as np
import matplotlib.pyplot as plt
from fourheap.fourheap import FourHeap
from fourheap.order import Order
from fourheap.constants import BUY, SELL
from event.event_queue import EventQueue
from market.market import Market
from fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from agent.hbl_agent import HBLAgent
from agent.zero_intelligence_agent import ZIAgent
from marketsim.simulator.sampled_arrival_simulator import SimulatorSampledArrival

NUM_AGENTS = 100
MEAN = 1e7
LAM = 1e-3
SIM_TIME = 1000
R = 0.05
SHOCK_VAR = 1e6
agents = {}


fundamental = LazyGaussianMeanReverting(mean=MEAN, final_time=SIM_TIME, r=R, shock_var=SHOCK_VAR)
markets = [Market(fundamental=fundamental, time_steps=SIM_TIME)]
for i in range(NUM_AGENTS - 1):
    agents[i] = ZIAgent(
                    agent_id=i,
                    market=markets[0],
                    q_max=20,
                    offset=12,
                    shade=[10, 50]
                )
agents[NUM_AGENTS - 1] = HBLAgent(agent_id=NUM_AGENTS-1, market=markets[0], q_max=100, offset=1, shade=[10,50], L=4)
                
sim = SimulatorSampledArrival(num_agents=NUM_AGENTS, sim_time=SIM_TIME, lam=0.1, mean=MEAN, r=R, shock_var=SHOCK_VAR, agents=agents, markets=markets)
sim.run()
print(agents[NUM_AGENTS - 1].HBL_MOVES, agents[NUM_AGENTS - 1].ORDERS)

# vals = sim.end_sim()
# y = [float(value) for value in vals.values()]
# plt.scatter([0]*len(y), y)  # Placing all points along the same x-axis position (0)
# plt.scatter([0], [float(list(vals.values())[-1])], color='red')  # Highlighting the point corresponding to key 99 with red color
# plt.xlabel('Value')
# plt.ylabel('Index')
# plt.title('1D Scatter Plot with Highlighted Point')
# plt.grid(True)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import time
from fourheap.fourheap import FourHeap
from fourheap.order import Order
from fourheap.constants import BUY, SELL
from event.event_queue import EventQueue
from market.market import Market
from fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from fundamental.mean_reverting import GaussianMeanReverting
from agent.hbl_agent_rewrite import HBLAgent
from agent.zero_intelligence_agent import ZIAgent
from marketsim.simulator.sampled_arrival_simulator import SimulatorSampledArrival
from marketsim.simulator.simulator import Simulator

NUM_AGENTS = 50
MEAN = 1e7
LAM = 1e-3
SIM_TIME = 2000
R = 0.05
SHOCK_VAR = 1e6
agents = {}


fundamental = LazyGaussianMeanReverting(mean=MEAN, final_time=SIM_TIME, r=R, shock_var=SHOCK_VAR)
markets = [Market(fundamental=fundamental, time_steps=SIM_TIME)]
for i in range(NUM_AGENTS - 1):
    agents[i] = ZIAgent(
                    agent_id=i,
                    market=markets[0],
                    q_max=100,
                    offset=12,
                    shade=[10, 30]
                )
    
#agents[NUM_AGENTS - 1] = HBLAgent(agent_id=NUM_AGENTS-1, market=markets[0], q_max=100, offset=1, shade=[10,30], L=2)
                
sim = SimulatorSampledArrival(num_agents=NUM_AGENTS, sim_time=SIM_TIME, lam=1e-2, mean=MEAN, r=R, shock_var=SHOCK_VAR, agents=agents, markets=markets)

start_time = time.time()
sim.run()

vals = sim.end_sim()
#print(agents[NUM_AGENTS - 1].HBL_MOVES, agents[NUM_AGENTS - 1].ORDERS)
print(len(sim.markets[0].matched_orders))
print("TIME OF EXEC", time.time() - start_time)
y = [float(value) for value in vals.values()]
plt.scatter([0]*len(y), y)  # Placing all points along the same x-axis position (0)
plt.scatter([0], [float(list(vals.values())[-1])], color='red')  # Highlighting the point corresponding to key 99 with red color
plt.xlabel('Ignore')
plt.ylabel('Final Surplus of Agent')
plt.title('HBL agent relative performance')
plt.grid(True)
plt.show()
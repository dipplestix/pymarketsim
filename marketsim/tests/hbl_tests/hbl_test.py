from tqdm import tqdm
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
# from agent.hbl_agent_rewrite import HBLAgent
from agent.zero_intelligence_agent import ZIAgent
from marketsim.simulator.sampled_arrival_simulator import SimulatorSampledArrival
from marketsim.simulator.simulator import Simulator

# NUM_AGENTS = 50
# MEAN = 1e7
# LAM = 1e-3
# SIM_TIME = 60000
# R = 0.05
# SHOCK_VAR = 1e6
# agents = {}

# sim = SimulatorSampledArrival(num_background_agents=100, sim_time=SIM_TIME, lam=1e-2, mean=MEAN, r=R, shock_var=SHOCK_VAR)

# start_time = time.time()
# sim.run()

# vals = sim.end_sim()
# #print(agents[NUM_AGENTS - 1].HBL_MOVES, agents[NUM_AGENTS - 1].ORDERS)
# print(len(sim.markets[0].matched_orders))
# print("TIME OF EXEC", time.time() - start_time)
# y = [float(value) for value in vals.values()]
# plt.scatter([0]*len(y), y)  # Placing all points along the same x-axis position (0)
# #plt.scatter([0], [float(list(vals.values())[-1])], color='red')  # Highlighting the point corresponding to key 99 with red color
# plt.xlabel('Ignore')
# plt.ylabel('Final Surplus of Agent')
# plt.title('ZI agent final performance')
# plt.grid(True)
# plt.show()
surpluses = []
valueAgents = []

for _ in tqdm(range(1000)):
    sim = SimulatorSampledArrival(num_background_agents=25, 
                                  sim_time=8000, 
                                  lam=5e-3, 
                                  mean=1e5, 
                                  r=0.05, 
                                  shock_var=5e6, 
                                  q_max=10,
                                  pv_var=5e6,
                                  shade=[250,500],
                                  hbl_agent=True)
    sim.run()
    fundamental_val = sim.markets[0].get_final_fundamental()
    values = []
    for agent_id in sim.agents:
        agent = sim.agents[agent_id]
        value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        # print(agent.cash, agent.position, agent.get_pos_value(), value)
        values.append(value)
    valueAgents.append(values)
    surpluses.append(sum(values)/len(values))

valueAgents = np.mean(valueAgents, axis = 0)
try:
    print(sim.agents[25].HBL_MOVES, sim.agents[25].ORDERS)
except:
    pass
num_agents = 26

input(valueAgents)
fig, ax = plt.subplots()
plt.scatter([0]*num_agents, valueAgents)  # Placing all points along the same x-axis position (0)
if num_agents == 26:
    plt.scatter([0], valueAgents[-1], color='red')
plt.xlabel('Ignore')
plt.show()

print(sum(surpluses)/len(surpluses)*25)
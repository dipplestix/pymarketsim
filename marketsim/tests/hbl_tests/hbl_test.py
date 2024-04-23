from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from simulator.sampled_arrival_simulator import SimulatorSampledArrival

surpluses = []
valueAgents = []

for i in tqdm(range(10000)):
    sim = SimulatorSampledArrival(num_background_agents=25, 
                                  sim_time=10000, 
                                  lam=5e-3, 
                                  mean=1e5, 
                                  r=0.05, 
                                  shock_var=1e5, 
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
    if i % 500 == 0:
        print(np.mean(valueAgents, axis = 0))
    surpluses.append(sum(values)/len(values))

valueAgents = np.mean(valueAgents, axis = 0)
num_agents = 26

input(valueAgents)
fig, ax = plt.subplots()
plt.scatter([0]*num_agents, valueAgents)  # Placing all points along the same x-axis position (0)
if num_agents == 26:
    plt.scatter([0], valueAgents[-1], color='red')
plt.xlabel('Ignore')
plt.show()

print(sum(surpluses)/len(surpluses)*num_agents)
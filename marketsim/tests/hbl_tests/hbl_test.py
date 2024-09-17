from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from agent.spoofer import SpoofingAgent
from simulator.sampled_arrival_simulator import SimulatorSampledArrival

SIM_TIME = 10000

valueAgentsNon = []

# random.seed(10)
for i in tqdm(range(8000)):
    sim = SimulatorSampledArrival(num_background_agents=24, 
                                  sim_time=SIM_TIME, 
                                  lam=2e-3, 
                                  mean=1e5, 
                                  r=0.05, 
                                  shock_var=1e4, 
                                  q_max=10,
                                  pv_var=5e6,
                                  shade=[250,500],
                                  hbl_agent=True,
                                  )
    sim.run()
    values = []
    fundamental_val = sim.markets[0].get_final_fundamental()
    for agent_id in sim.agents:
        agent = sim.agents[agent_id]
        value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        # print(agent.cash, fundamental_val, agent.position, agent.get_pos_value(), value)
        # input()
        values.append(value)
    valueAgentsNon.append(values)

    if i % 100 == 0:
        valueAgents = np.mean(valueAgentsNon, axis = 0)
        print(valueAgents)
        plt.figure()
        # num_agents = [j for j in range(15)]
        # plotNon = np.mean(valueAgentsNon, axis = 0)
        # barsNon = plt.bar(num_agents, plotNon, color='g', edgecolor='grey', label='Nonspoof')
        # plt.legend()
        # plt.title('Surpluses')
        # plt.xlabel('Agent')
        # plt.ylabel('Values')
        # plt.savefig('spoofer_exps/hbl_prelim/15_3_v2/{}.png'.format(i))
        plt.close()

from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from marketsim.simulator.sampled_arrival_simulator import SimulatorSampledArrival
from marketsim.wrappers.SP_wrapper import SPEnv
from marketsim.private_values.private_values import PrivateValues
import torch.distributions as dist
import torch

SIM_TIME = 10000

valueAgentsSpoof = []
valueAgentsNon = []
diffs = []
fundamentals = []
avg_spoof_most_recent_trade = []
avg_non_most_recent_trade = []

def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps

for i in tqdm(range(10000)):
    '''
    Intended behavior: HBL agent surplus in spoofing environment is significantly lower than the HBL agent surplus in market with no spoofer.
    '''
    sim = SimulatorSampledArrival(num_background_agents=15, 
                                  sim_time=SIM_TIME, 
                                  lam=5e-3, 
                                  mean=1e5, 
                                  r=0.05, 
                                  shock_var=5e6, 
                                  q_max=10,
                                  pv_var=5e6,
                                  shade=[250,500],
                                  hbl_agent=True,
                                  fundamental = fundamental)

    normalizers = {"fundamental": 1e5, "reward":1e4, "min_order_val": 1e5, "invt": 10, "cash": 1e7}
    env = SPEnv(num_background_agents=15,
                sim_time=SIM_TIME,
                lam=5e-3,
                lamSP=5e-2,
                mean=1e5,
                r=0.05,
                shock_var=5e6,
                q_max=10,
                pv_var=5e6,
                shade=[250,500],
                normalizers=normalizers,
                fundamental = fundamental)

    obs, info = env.reset()
    # random.seed(8)
    while env.time < SIM_TIME:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
    # random.seed(8)
    sim.run()

    def estimate_fundamental(t):
        mean = 1e5
        r = 0.05
        T = 10000
        val = sim.markets[0].fundamental.get_value_at(t)
        rho = (1 - r) ** (T - t)

        estimate = (1 - rho) * mean + rho * val
        return estimate

    diffs.append(np.subtract(np.array(list(env.most_recent_trade.values())),np.array(list(sim.most_recent_trade.values()))))
    fundamental = []
    for j in range(0,SIM_TIME + 1):
        fundamental.append(estimate_fundamental(j))
    fundamentals.append(fundamental)
    avg_spoof_most_recent_trade.append(list(env.most_recent_trade.values()))
    avg_non_most_recent_trade.append(list(sim.most_recent_trade.values()))
    
    fundamental_val = sim.markets[0].get_final_fundamental()
    valuesSpoof = []
    valuesNon = []
    for agent_id in env.agents:
        agent = env.agents[agent_id]
        value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        valuesSpoof.append(value)
    agent = env.spoofer
    value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
    valuesSpoof.append(value)

    for agent_id in sim.agents:
        agent = sim.agents[agent_id]
        value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        valuesNon.append(value)
    #placeholder for additional spoofer
    valuesNon.append(0)
    valueAgentsSpoof.append(valuesSpoof)
    valueAgentsNon.append(valuesNon)

    if (i + 1) % 2000 == 0:
        plt.figure()
        plt.plot([i for i in range(0, SIM_TIME+1)], np.mean(avg_spoof_most_recent_trade,axis=0), label="spoof")
        plt.plot([i for i in range(0, SIM_TIME+1)], np.mean(avg_non_most_recent_trade,axis=0), linestyle='--',label="Nonspoof")
        plt.legend()
        plt.xlabel('Timesteps')
        plt.ylabel('Last matched order price')
        plt.title('Spoof v Nonspoof last matched trade price')
        plt.savefig('spoofer_exps/market_exps_v3/{}_matched.png'.format(i))
        plt.close()

        plt.figure()
        plt.plot([i for i in range(0, SIM_TIME+1)], np.mean(diffs,axis=0))
        plt.title('Spoofer diff - RUNNING AVERAGE')
        plt.xlabel('Timesteps')
        plt.ylabel('Difference')

        # Save the figure
        plt.savefig('spoofer_exps/market_exps_v3/{}_matched_diffs.png'.format(i))
        plt.close()

        print("FINAL TIMESTEP FUNDAMENTAL")
        print(sim.markets[0].get_final_fundamental())
        print(env.markets[0].get_final_fundamental())
        input()
        plt.figure()
        bar_width = 0.35
        num_agents = [j for j in range(16)]
        num_agent_non= [x + bar_width for x in num_agents]
        plotSpoof = np.mean(valueAgentsSpoof, axis = 0)
        plotNon = np.mean(valueAgentsNon, axis = 0)
        barsSpoof = plt.bar(num_agents, plotSpoof, color='b', width=bar_width, edgecolor='grey', label='Spoof')
        barsNon = plt.bar(num_agent_non, plotNon, color='g', width=bar_width, edgecolor='grey', label='Nonspoof')
        plt.legend()
        print("Spoofer environment surplus:")
        print(valueAgentsSpoof)
        print("Nonspoofer environment surplus:")
        print(valueAgentsNon)

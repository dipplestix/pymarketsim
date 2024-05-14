from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from simulator.sampled_arrival_simulator import SimulatorSampledArrival
from wrappers.SP_wrapper import SPEnv
from marketsim.private_values.private_values import PrivateValues
import torch.distributions as dist
import torch
from fundamental.mean_reverting import GaussianMeanReverting

SIM_TIME = 9000

surpluses = []
valueAgents = []
diffs = []
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps


# random.seed(10)
for i in tqdm(range(10000)):
    a = [PrivateValues(10,5e6) for _ in range(0,15)]
    sampled_arr = sample_arrivals(5e-3,SIM_TIME)
    fundamental = GaussianMeanReverting(mean=1e5, final_time=SIM_TIME + 1, r=0.05, shock_var=5e6)
    random.seed(12)
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
                                  pvalues = a,
                                  sampled_arr=sampled_arr,
                                  fundamental = fundamental)

    normalizers = {"fundamental": 1e5, "reward":1e4, "min_order_val": 1e5, "invt": 10, "cash": 1e7}
    random.seed(12)
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
                pvalues = a,
                sampled_arr=sampled_arr,
                fundamental = fundamental)

    obs, info = env.reset()
    random.seed(8)
    while env.time < SIM_TIME:
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
    random.seed(8)
    sim.run()
    diffs.append(np.subtract(np.array(list(env.most_recent_trade.values())),np.array(list(sim.most_recent_trade.values()))))

    if i % 1 == 0:
        plt.figure()
        plt.plot([i for i in range(0, 9001)], np.mean(diffs,axis=0))
        plt.title('Spoofer diff')
        plt.xlabel('Timesteps')
        plt.ylabel('Difference')

        # Save the figure
        plt.savefig('spoofer_diff_sim_{}.png'.format(i))


    # fundamental_val = sim.markets[0].get_final_fundamental()
    # values = []
    # for agent_id in sim.agents:
    #     agent = sim.agents[agent_id]
    #     value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
    #     # print(agent.cash, agent.position, agent.get_pos_value(), value)
    #     values.append(value)
    # valueAgents.append(values)
    # if i % 500 == 0:
    #     print(np.mean(valueAgents, axis = 0))
    # surpluses.append(sum(values)/len(values))

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
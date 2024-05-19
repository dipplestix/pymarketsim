from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from simulator.sampled_arrival_simulator import SimulatorSampledArrival
from wrappers.SP_wrapper import SPEnv
from private_values.private_values import PrivateValues
import torch.distributions as dist
import torch
from fundamental.mean_reverting import GaussianMeanReverting
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

SIM_TIME = 10000
TOTAL_ITERS = 3
NUM_AGENTS = 15
LEARNING = False
graphVals = 1
printVals = 100

valueAgentsSpoof = []
valueAgentsNon = []
diffs = []
env_trades = []
sim_trades = []
sell_above_best_avg = []

path = "spoofer_exps/baseline_fixed/4"
print("GRAPH SAVE PATH", path)

normalizers = {"fundamental": 1e5, "reward":1e4, "min_order_val": 1e5, "invt": 10, "cash": 1e7}
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps

if LEARNING:
    a = [PrivateValues(10,5e6) for _ in range(0,NUM_AGENTS)]
    sampled_arr = sample_arrivals(5e-3,SIM_TIME)
    fundamental = GaussianMeanReverting(mean=1e5, final_time=SIM_TIME + 1, r=0.05, shock_var=5e6)
    env = SPEnv(num_background_agents=NUM_AGENTS,
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
                fundamental = fundamental,
                learning = LEARNING)

    model = SAC("MlpPolicy", env,  verbose=1)
    model.learn(total_timesteps=1000)

# random.seed(10)
for i in tqdm(range(TOTAL_ITERS)):
    a = [PrivateValues(10,5e6) for _ in range(0,NUM_AGENTS)]
    sampled_arr = sample_arrivals(5e-3,SIM_TIME)
    fundamental = GaussianMeanReverting(mean=1e5, final_time=SIM_TIME + 1, r=0.05, shock_var=5e6)
    random.seed(12)
    sim = SimulatorSampledArrival(num_background_agents=NUM_AGENTS, 
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

    
    random.seed(12)
    env = SPEnv(num_background_agents=NUM_AGENTS,
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
                fundamental = fundamental,
                learning = LEARNING)

    observation, info = env.reset()
    random.seed(8)
    while env.time < SIM_TIME:
        action = env.action_space.sample()  # this is where you would insert your policy
        # action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
    random.seed(8)
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
    env_trades.append(list(env.most_recent_trade.values()))
    sim_trades.append(list(sim.most_recent_trade.values()))
    sell_above_best_avg.append(np.mean(env.sell_above_best))
    fundamentalEvol = []
    for j in range(0,SIM_TIME + 1):
        fundamentalEvol.append(estimate_fundamental(j))
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
        # print(agent.cash, fundamental_val, agent.position, agent.get_pos_value(), value)
        # input()
        valuesNon.append(value)
    #placeholder for additional spoofer
    valuesNon.append(0)
    valueAgentsSpoof.append(valuesSpoof)
    valueAgentsNon.append(valuesNon)

    if i % graphVals == 0:        
        x_axis = [i for i in range(0, SIM_TIME+1)]

        plt.figure()
        plt.plot(x_axis, np.nanmean(diffs,axis=0))
        plt.title('Spoofer diff - RUNNING AVERAGE')
        plt.xlabel('Timesteps')
        plt.ylabel('Difference')

        # Save the figure
        plt.savefig(path + '/{}_spoofer_diff_sim.png'.format(i))
        plt.close()

        plt.figure()
        plt.plot(x_axis, np.nanmean(env_trades, axis=0), label="spoof")
        plt.plot(x_axis, np.nanmean(sim_trades, axis=0), linestyle='dotted',label="Nonspoof")
        plt.legend()
        plt.xlabel('Timesteps')
        plt.ylabel('Last matched order price')
        plt.title('Spoof v Nonspoof last matched trade price - AVERAGED')
        plt.savefig(path + '/{}_AVG_matched_order_price.png'.format(i))
        plt.close()
        
        plt.figure()
        plt.plot(x_axis, list(env.most_recent_trade.values()), label="spoof", linestyle="dotted")
        plt.plot(x_axis, list(sim.most_recent_trade.values()), linestyle='--',label="Nonspoof")
        plt.legend()
        plt.xlabel('Timesteps')
        plt.ylabel('Last matched order price')
        plt.title('Spoof v Nonspoof last matched trade price - NOT AVERAGED')
        plt.savefig(path + '/{}_NONAVG_matched_order_price.png'.format(i))
        plt.close()

        plt.figure()
        a = list(env.spoof_orders.values())
        b = list(env.sell_orders.values())
        plt.plot(x_axis, list(env.spoof_orders.values()), label="spoof", color="magenta", zorder=10)
        plt.plot(x_axis, list(env.best_buys.values()), label="best buys", linestyle="--", color="cyan")
        plt.plot(x_axis, list(env.best_asks.values()), label="best asks", linestyle="--", color="yellow")
        # plt.plot(x_axis, fundamentalEvol, label="fundamental", linestyle="dotted", zorder=0)
        plt.plot(x_axis, list(env.sell_orders.values()), label="sell orders", color="black")
        plt.legend()
        plt.xlabel('Timesteps')
        plt.ylabel('Price')
        plt.title('Price comparisons of spoofer orders - NOT AVERAGED')
        plt.savefig(path + '/{}_spoofer_orders.png'.format(i))
        plt.close()

        plt.figure()
        plt.plot([i for i in range(len(env.sell_above_best))], env.sell_above_best, label="sells")
        plt.xlabel('Timesteps?')
        plt.ylabel('Spoof sell - best ask')
        plt.title('Spoof sell - best ask - NOT AVERAGED')
        plt.savefig(path + '/{}_sell_above_ask.png'.format(i))
        plt.close()

        plt.figure()
        bar_width = 0.35
        num_agents = [j for j in range(NUM_AGENTS + 1)]
        num_agent_non= [x + bar_width for x in num_agents]
        plotSpoof = np.nanmean(valueAgentsSpoof, axis = 0)
        plotNon = np.nanmean(valueAgentsNon, axis = 0)
        barsSpoof = plt.bar(num_agents, plotSpoof, color='b', width=bar_width, edgecolor='grey', label='Spoof')
        barsNon = plt.bar(num_agent_non, plotNon, color='g', width=bar_width, edgecolor='grey', label='Nonspoof')
        plt.legend()
        # for bar, values in zip(barsSpoof, plotSpoof):
        #     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), values, ha='center', va='bottom')
        # for bar, values in zip(barsNon, plotNon):
        #     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), values, ha='center', va='bottom')        
        plt.title('Surplus Comparison Spoof/Nonspoof')
        plt.xlabel('Agent')
        plt.ylabel('Values')
        plt.xticks([r + bar_width/2 for r in range(len(num_agents))], num_agents)
        plt.savefig(path + '/{}_surpluses_sim.png'.format(i))
        plt.close()
    if i % printVals == 0:
        print("SPOOFER")
        print(plotSpoof)
        print("NONSPOOFER")
        print(plotNon)
        print("\n SPOOFER POSITION TRACK")
        print(env.spoofer.position)


plotSpoof = np.nanmean(valueAgentsSpoof, axis = 0)
plotNon = np.nanmean(valueAgentsNon, axis = 0)
print("SPOOFER")
print(plotSpoof)
print("NONSPOOFER")
print(plotNon)
print("\n SPOOFER POSITION TRACK")
print(env.spoofer.position)


x_axis = [i for i in range(0, SIM_TIME+1)]
plt.figure()
plt.plot(x_axis, np.nanmean(diffs,axis=0))
plt.title('Spoofer diff - RUNNING AVERAGE')
plt.xlabel('Timesteps')
plt.ylabel('Difference')
# Save the figure
plt.savefig(path + '/{}_spoofer_diff_sim.png'.format(TOTAL_ITERS))
plt.close()

plt.figure()
plt.plot(x_axis, np.nanmean(env_trades, axis=0), label="spoof")
plt.plot(x_axis, np.nanmean(sim_trades, axis=0), linestyle='dotted',label="Nonspoof")
plt.legend()
plt.xlabel('Timesteps')
plt.ylabel('Last matched order price')
plt.title('Spoof v Nonspoof last matched trade price - AVERAGED')
plt.savefig(path + '/{}_AVG_matched_order_price.png'.format(TOTAL_ITERS))
plt.close()

plt.figure()
plt.plot(x_axis, list(env.most_recent_trade.values()), label="spoof", linestyle="dotted")
plt.plot(x_axis, list(sim.most_recent_trade.values()), linestyle='--',label="Nonspoof")
plt.legend()
plt.xlabel('Timesteps')
plt.ylabel('Last matched order price')
plt.title('Spoof v Nonspoof last matched trade price - NOT AVERAGED')
plt.savefig(path + '/{}_NONAVG_matched_order_price.png'.format(TOTAL_ITERS))
plt.close()

plt.figure()
a = list(env.spoof_orders.values())
b = list(env.sell_orders.values())
plt.plot(x_axis, list(env.spoof_orders.values()), label="spoof", color="magenta", zorder=10)
plt.plot(x_axis, list(env.best_buys.values()), label="best buys", linestyle="--", color="cyan")
plt.plot(x_axis, list(env.best_asks.values()), label="best asks", linestyle="--", color="yellow")
# plt.plot(x_axis, fundamentalEvol, label="fundamental", linestyle="dotted", zorder=0)
plt.plot(x_axis, list(env.sell_orders.values()), label="sell orders", color="black")
plt.legend()
plt.xlabel('Timesteps')
plt.ylabel('Price')
plt.title('Price comparisons of spoofer orders - NOT AVERAGED')
plt.savefig(path + '/{}_spoofer_orders.png'.format(TOTAL_ITERS))
plt.close()

plt.figure()
bar_width = 0.35
num_agents = [j for j in range(NUM_AGENTS + 1)]
num_agent_non= [x + bar_width for x in num_agents]
plotSpoof = np.nanmean(valueAgentsSpoof, axis = 0)
plotNon = np.nanmean(valueAgentsNon, axis = 0)
barsSpoof = plt.bar(num_agents, plotSpoof, color='b', width=bar_width, edgecolor='grey', label='Spoof')
barsNon = plt.bar(num_agent_non, plotNon, color='g', width=bar_width, edgecolor='grey', label='Nonspoof')
plt.legend()
# for bar, values in zip(barsSpoof, plotSpoof):
#     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), values, ha='center', va='bottom')
# for bar, values in zip(barsNon, plotNon):
#     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), values, ha='center', va='bottom')        
plt.title('Surplus Comparison Spoof/Nonspoof')
plt.xlabel('Agent')
plt.ylabel('Values')
plt.xticks([r + bar_width/2 for r in range(len(num_agents))], num_agents)
plt.savefig(path + '/{}_surpluses_sim.png'.format(TOTAL_ITERS))
plt.close()

input(sell_above_best_avg)
plt.figure()
plt.plot([i for i in range(TOTAL_ITERS)], sell_above_best_avg, label="sells")
plt.xlabel('Timesteps?')
plt.ylabel('Spoof sell - best ask')
plt.title('Spoof sell - best ask - AVERAGED')
plt.savefig(path + '/OVERALL_sell_above_ask.png'.format(TOTAL_ITERS))
plt.close()

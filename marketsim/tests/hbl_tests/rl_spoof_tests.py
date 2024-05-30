from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from fourheap.constants import BUY, SELL
from simulator.sampled_arrival_simulator import SimulatorSampledArrival
from wrappers.SP_wrapper import SPEnv
from wrappers.Paired_SP_wrapper import NonSPEnv
from private_values.private_values import PrivateValues
import torch.distributions as dist
import torch
from fundamental.mean_reverting import GaussianMeanReverting
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

SIM_TIME = 10000
TOTAL_ITERS = 10000
NUM_AGENTS = 25
LEARNING = False
graphVals = 200
printVals = 500

valueAgentsSpoof = []
valueAgentsNon = []
diffs = []
env_trades = []
sim_trades = []
sell_above_best_avg = []
spoofer_position = []
nonspoofer_position = []

path = "spoofer_baseline_exps/5e4_arrival/500"
print("GRAPH SAVE PATH", path)

normalizers = {"fundamental": 1e5, "reward":1e3, "min_order_val": 1e5, "invt": 10, "cash": 1e7}
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps

def make_env(spEnv: SPEnv):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init() -> SPEnv:
        env = spEnv
        env.reset()
        return spEnv

    return _init

def run():
    if LEARNING:
        learningEnv = SPEnv(num_background_agents=NUM_AGENTS,
                    sim_time=SIM_TIME,
                    lam=5e-3,
                    lamSP=5e-2,
                    mean=1e5,
                    r=0.05,
                    shock_var=1e6,
                    q_max=10,
                    pv_var=5e6,
                    shade=[250,500],
                    normalizers=normalizers,
                    learning = LEARNING,
                    analytics = False)
        
        num_cpu = 1  # Number of processes to use
        # Create the vectorized environment
        if num_cpu == 1:
            spEnv = make_vec_env(make_env(learningEnv), n_envs=1, vec_env_cls=DummyVecEnv)
        else:
            spEnv = make_vec_env(make_env(learningEnv), n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
        # spEnv = SubprocVecEnv([make_env(env) for _ in range(num_cpu)])
        
        # We collect 4 transitions per call to `ènv.step()`
        # and performs 2 gradient steps per call to `ènv.step()`
        # if gradient_steps=-1, then we would do 4 gradients steps per call to `ènv.step()`
        model = SAC("MlpPolicy", spEnv, train_freq=1, gradient_steps=-1, verbose=1)
        model.learn(total_timesteps=1e6, progress_bar=True)

    random.seed(10)
    for i in tqdm(range(TOTAL_ITERS)):
        random_seed = [random.randint(0,100000) for _ in range(10000)]

        a = [PrivateValues(10,5e6) for _ in range(0,NUM_AGENTS + 1)]
        sampled_arr = sample_arrivals(5e-3,SIM_TIME)
        spoofer_arrivals = sample_arrivals(5e-2,SIM_TIME)
        fundamental = GaussianMeanReverting(mean=1e5, final_time=SIM_TIME + 1, r=0.05, shock_var=5e5)
        random.seed(12)
        sim = NonSPEnv(num_background_agents=NUM_AGENTS,
                    sim_time=SIM_TIME,
                    lam=5e-4,
                    lamSP=5e-2,
                    mean=1e5,
                    r=0.05,
                    shock_var=1e6,
                    q_max=10,
                    pv_var=5e6,
                    shade=[250,500],
                    pvalues = a,
                    sampled_arr=sampled_arr,
                    spoofer_arrivals=spoofer_arrivals,
                    fundamental = fundamental,
                    analytics=True,
                    random_seed = random_seed
                    )
        observation, info = sim.reset()
        random.seed(12)
        env = SPEnv(num_background_agents=NUM_AGENTS,
                    sim_time=SIM_TIME,
                    lam=5e-4,
                    lamSP=5e-2,
                    mean=1e5,
                    r=0.05,
                    shock_var=1e6,
                    q_max=10,
                    pv_var=5e6,
                    shade=[250,500],
                    normalizers=normalizers,
                    pvalues = a,
                    sampled_arr=sampled_arr,
                    spoofer_arrivals=spoofer_arrivals,
                    fundamental = fundamental,
                    learning = False,
                    learnedActions = True,
                    analytics = True,
                    random_seed = random_seed
                    )

        observation, info = env.reset()

        random.seed(8)
        while sim.time < SIM_TIME:
            sim.step()

        random.seed(8)
        while env.time < SIM_TIME:
            if LEARNING:
                action, _states = model.predict(observation, deterministic=True)
            else:
                action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)

        def estimate_fundamental(t):
            mean = 1e5
            r = 0.05
            T = 10000
            val = sim.markets[0].fundamental.get_value_at(t)
            rho = (1 - r) ** (T - t)

            estimate = (1 - rho) * mean + rho * val
            return estimate

        diffs.append(np.subtract(np.array(list(env.most_recent_trade.values())),np.array(list(sim.most_recent_trade.values()))))
        # input(list(env.most_recent_trade.values()))
        # input(list(sim.most_recent_trade.values()))
        env_trades.append(list(env.most_recent_trade.values()))
        sim_trades.append(list(sim.most_recent_trade.values()))
        sell_above_best_avg.append(np.mean(env.sell_above_best))
        fundamentalEvol = []
        spoofer_position.append(list(env.spoofer_quantity.values()))
        nonspoofer_position.append(list(sim.spoofer_quantity.values()))
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
            plt.plot(x_axis, np.nanmean(spoofer_position, axis=0), label="Position")
            plt.plot(x_axis, np.nanmean(nonspoofer_position, axis=0), label="Position_Non")
            plt.xlabel('Timesteps')
            plt.ylabel('Position')
            plt.title('AVERAGED - Position of Spoofer Over Time')
            plt.savefig(path + '/{}_AVG_spoofer_position.png'.format(i))
            plt.close()


            plt.figure()
            plt.hist(range(len(list(env.trade_volume.values()))), bins=len(list(env.trade_volume.values()))//100, weights=list(env.trade_volume.values()), edgecolor='black')
            plt.xlabel('Timesteps')
            plt.ylabel('# trades')
            plt.title('Spoof trade volume')
            plt.savefig(path + '/{}_NONAVG_trade_volume_spoof.png'.format(i))
            plt.close()
            
            plt.figure()
            plt.hist(range(len(list(sim.trade_volume.values()))), bins=len(list(sim.trade_volume.values()))//100, weights=list(sim.trade_volume.values()), edgecolor='black')
            plt.xlabel('Timesteps')
            plt.ylabel('# trades')
            plt.title('Nonspoof trade volume')
            plt.savefig(path + '/{}_NONAVG_trade_volume_nonspoof.png'.format(i))
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
            plt.bar(num_agents, plotSpoof, color='b', width=bar_width, edgecolor='grey', label='Spoof')
            plt.bar(num_agent_non, plotNon, color='g', width=bar_width, edgecolor='grey', label='Nonspoof')
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


    plt.figure()
    plt.plot(x_axis, list(env.spoofer_quantity.values()), label="Position")
    plt.xlabel('Timesteps')
    plt.ylabel('Position')
    plt.title('Position of Spoofer Over Time')
    plt.savefig(path + '/{}_NONAVG_spoofer_position.png'.format(i))
    plt.close()

    plt.figure()
    plt.plot(x_axis, list(env.spoofer_value.values()), label="Value")
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.title('Value of Spoofer Over Time')
    plt.savefig(path + '/{}_NONAVG_spoofer_value.png'.format(i))
    plt.close()


    plt.figure()
    plt.plot([i for i in range(TOTAL_ITERS)], sell_above_best_avg, label="sells")
    plt.xlabel('Timesteps?')
    plt.ylabel('Spoof sell - best ask')
    plt.title('Spoof sell - best ask - AVERAGED')
    plt.savefig(path + '/OVERALL_sell_above_ask.png'.format(TOTAL_ITERS))
    plt.close()

if __name__ == "__main__":
    run()
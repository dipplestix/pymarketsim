from tqdm import tqdm
import random
import numpy as np
import gzip
import matplotlib.pyplot as plt
import time
import sys
import os, shutil
import torch as th
import pickle
from fourheap.constants import BUY, SELL
from simulator.sampled_arrival_simulator import SimulatorSampledArrival
from wrappers.SP_wrapper import SPEnv
from wrappers.PairedMMSP_wrapper import PairedMMSPEnv
from wrappers.StaticMMSP_wrapper import MMSPEnv
from private_values.private_values import PrivateValues
import torch.distributions as dist
import torch
from fundamental.mean_reverting import GaussianMeanReverting
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC, DDPG, TD3, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from custom_callback import SaveOnBestTrainingRewardCallback
import torch

SIM_TIME = 10000
TOTAL_ITERS = 8000
NUM_AGENTS = 25
LEARNING = False
LEARNING_ACTIONS = False
PAIRED = True

graphVals = 300
printVals = 300

BASE_PATH = "None"
PICKLE_PATH = "None"
CALLBACK_LOG_DIR = "None"
LEARNING_GRAPH_PATH = "None"

def create_dirs(base_path):
    global BASE_PATH
    global PICKLE_PATH
    global CALLBACK_LOG_DIR
    global LEARNING_GRAPH_PATH
    BASE_PATH = base_path
    PICKLE_PATH = base_path + "/pickle"
    CALLBACK_LOG_DIR = base_path + "/c"
    LEARNING_GRAPH_PATH = base_path + "/learning_graphs"

    for dir in [BASE_PATH, CALLBACK_LOG_DIR, LEARNING_GRAPH_PATH, PICKLE_PATH]:
    # for dir in [PICKLE_PATH]:
        try:
            shutil.rmtree(dir)
            print("Removed {}".format(dir))
        except:
            print("{} doesn't exist".format(dir))
            print("Creating {}".format(dir))
        os.makedirs(dir, exist_ok=True)

    print("CALLBACK PATH", CALLBACK_LOG_DIR)

mm_params = {"xi": 100, "omega": 256, "K": 8}
arrival_rates = {"lam":2e-3, "lamSP": 6e-3, "lamMM": 0.035}
market_params = {"r":0.05, "mean": 1e5, "shock_var": 1e4, "pv_var": 5e6}
normalizers = {"fundamental": 1e5, "reward":1e2, "min_order_val": 1e5, "invt": 10, "cash": 1e6}
# normalizers = {"fundamental": 1, "reward":1, "min_order_val": 1, "invt": 1, "cash": 1}

def append_pickle(data, file_path):
    """
    Append data to a pickle file. If the file does not exist, it creates a new one.
    """
    file_path += ".pkl"
    if os.path.exists(file_path):
        # If the file exists, open it in append mode
        with gzip.open(file_path, 'ab') as f:
            pickle.dump(data, f)
    else:
        # If the file does not exist, create it and write the data
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(data, f)


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

def run(data_path, load_path):
    create_dirs(data_path)
    valueAgentsSpoof = []
    env_trades = []
    env_sell_orders = []
    env_spoof_orders =[]
    env_est_fund = []
    spoofer_surplus = []
    spoofer_position = []
    env_best_buys = []
    env_best_asks = []
    if LEARNING:
        print("GPU =", torch.cuda.is_available())
        learningEnv = MMSPEnv(num_background_agents=NUM_AGENTS,
                    sim_time=SIM_TIME,
                    lam=arrival_rates["lam"],
                    lamSP=arrival_rates["lamSP"],
                    lamMM=arrival_rates["lamMM"],
                    mean=market_params["mean"],
                    r=market_params["r"],
                    shock_var=market_params["shock_var"],
                    q_max=10,
                    pv_var=market_params["pv_var"],
                    shade=[250,500],
                    normalizers=normalizers,
                    learning = LEARNING,
                    learnedActions = LEARNING_ACTIONS,
                    analytics = True,
                    xi=mm_params["xi"], # rung size
                    omega=mm_params["omega"], #spread
                    K=mm_params["K"],
                    order_size=1, # the size of regular order: NEED TUNING
                    spoofing_size=200, # the size of spoofing order: NEED TUNING
                    learning_graphs_path=LEARNING_GRAPH_PATH
                    )
        
        num_cpu = 1  # Number of processes to use
        # Create the vectorized environment
        if num_cpu == 1:
            spEnv = make_vec_env(make_env(learningEnv), n_envs=1, monitor_dir=CALLBACK_LOG_DIR, vec_env_cls=DummyVecEnv)
        else:
            spEnv = make_vec_env(make_env(learningEnv), n_envs=num_cpu, monitor_dir=CALLBACK_LOG_DIR, vec_env_cls=SubprocVecEnv)
        # spEnv = SubprocVecEnv([make_env(env) for _ in range(num_cpu)])
        # check_env(learningEnv)
        # print("DONE")
        # input()
        # Create the callback: check every 1000 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=CALLBACK_LOG_DIR)
        # We collect 4 transitions per call to `ènv.step()`
        # and performs 2 gradient steps per call to `ènv.step()`
        # if gradient_steps=-1, then we would do 4 gradients steps per call to `ènv.step()`
        # n_actions = spEnv.action_space.shape[-1]
        # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
        model = PPO("MlpPolicy", spEnv, verbose=1, device="cuda", clip_range=0.1)
        # model = RecurrentPPO("MlpLstmPolicy", spEnv, verbose=1, device="cuda", clip_range=0.1)
        model.learn(total_timesteps=4e5, progress_bar=True, callback=callback)
        # policy_kwargs=dict(net_arch=dict(pi=[128,128], vf=[512,512]))
        # print(callback.cumulative_window_rewards)
        print("Loading best model...")
        model = PPO.load(os.path.join(data_path, "c", "best_model.zip"))
        print("LOADED!", model)

    for i in tqdm(range(TOTAL_ITERS)):
        random_seed = [random.randint(0,100000) for _ in range(SIM_TIME)]
        privateValues = [PrivateValues(10,market_params["pv_var"]) for _ in range(0,NUM_AGENTS - 1)]
        sampled_arr = sample_arrivals(arrival_rates["lam"],SIM_TIME)
        spoofer_arrivals = sample_arrivals(arrival_rates["lamSP"],SIM_TIME)
        MM_arrivals = sample_arrivals(arrival_rates["lamMM"],SIM_TIME)
        fundamental = GaussianMeanReverting(mean=market_params["mean"], final_time=SIM_TIME + 1, r=market_params["r"], shock_var=market_params["shock_var"])
        env = MMSPEnv(num_background_agents=NUM_AGENTS,
                    sim_time=SIM_TIME,
                    lam=arrival_rates["lam"],
                    lamSP=arrival_rates["lamSP"],
                    lamMM=arrival_rates["lamMM"],
                    mean=market_params["mean"],
                    r=market_params["r"],
                    shock_var=market_params["shock_var"],
                    q_max=10,
                    pv_var=market_params["pv_var"],
                    shade=[250,500],
                    normalizers=normalizers,
                    pvalues = privateValues,
                    sampled_arr=sampled_arr,
                    spoofer_arrivals=spoofer_arrivals,
                    MM_arrivals=MM_arrivals,
                    fundamental = fundamental,
                    learning = False,
                    learnedActions = LEARNING_ACTIONS,
                    analytics = True,
                    random_seed = random_seed,
                    xi=mm_params["xi"], # rung size
                    omega=mm_params["omega"], #spread
                    K=mm_params["K"],
                    order_size=1, # the size of regular order: NEED TUNING
                    spoofing_size=200, # the size of spoofing order: NEED TUNING
                    )

        observation, info = env.reset()

        random.seed(8)
        while env.time < SIM_TIME:
            if LEARNING_ACTIONS:
                action, _states = model.predict(observation, deterministic=True)
            else:
                action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)

        env_trades.append(list(env.most_recent_trade.values()))
        spoofer_surplus.append(list(env.spoof_activity.values()))
        env_best_buys.append(list(env.best_buys.values()))
        env_best_asks.append(list(env.best_asks.values()))
        spoofer_position.append(list(env.spoofer_quantity.values()))
        env_est_fund.append(list(env.est_funds.values()))
        env_spoof_orders.append(list(env.spoof_orders.values()))
        env_sell_orders.append(list(env.sell_orders.values()))
        fundamental_val = env.markets[0].get_final_fundamental()
        
        valuesSpoof = []
        
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
            valuesSpoof.append(value)
        
        mm = env.MM
        value = mm.position * fundamental_val + mm.cash
        valuesSpoof.append(value)

        agent = env.spoofer
        value = agent.position * fundamental_val + agent.cash
        valuesSpoof.append(value)
        valueAgentsSpoof.append(valuesSpoof)

        if (i + 1) % 2000 == 0:
            append_pickle(np.array(valueAgentsSpoof), PICKLE_PATH + "/values_env")
            append_pickle(np.array(env_trades), PICKLE_PATH + "/trades_env")
            append_pickle(np.array(spoofer_position), PICKLE_PATH + "/position_env")
            append_pickle(np.array(spoofer_surplus), PICKLE_PATH + "/surplus_env")
            append_pickle(np.array(env_est_fund), PICKLE_PATH + "/env_est_funds")
            append_pickle(np.array(env_sell_orders), PICKLE_PATH + "/env_sell_orders")
            append_pickle(np.array(env_spoof_orders), PICKLE_PATH + "/env_spoof_orders")
            append_pickle(np.array(env_best_buys), PICKLE_PATH + "/env_best_buys")
            append_pickle(np.array(env_best_asks), PICKLE_PATH + "/env_best_asks")
            print("DATA SAVED {}".format(i))
            valueAgentsSpoof = []
            env_trades = []
            spoofer_position = []
            spoofer_surplus = []
            env_est_fund = []
            env_sell_orders = []
            env_spoof_orders = []
            env_best_buys = []
            env_best_asks = []
            
if __name__ == "__main__":
    data_path = sys.argv[1]
    load_path = sys.argv[2]
    run(data_path, load_path)
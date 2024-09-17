from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import gzip
from fourheap.constants import BUY, SELL


BASE_PATH = "no_spoof/"
PICKLE_PATH = BASE_PATH + "/pickle"

def load_pickle(file_path):
    """
    Load all objects from a pickle file.
    """
    file_path += ".pkl"
    data = []
    if os.path.exists(file_path):
        with gzip.open(file_path, 'rb') as f:
            while True:
                try:
                    if len(data) == 0:
                        data = list(pickle.load(f))
                    else:
                        data.extend(list(pickle.load(f)))
                except EOFError:
                    break
    return data

# def loop_all_files(data_path, avgValueAgents, avgValueNon, avg_env_trades, 
#                     avg_sim_trades, avg_spoof_position, avg_non_position, avg_diffs, avg_spoofer_surplus, 
#                     avg_nonspoofer_surplus, avg_env_est_fund, avg_env_sell_orders, avg_sim_sell_orders, 
#                     avg_env_spoof_orders, avg_sim_spoof_orders, avg_env_best_buys, avg_sim_best_buys, 
#                     avg_env_best_asks, avg_sim_best_asks, avg_env_buy_below, avg_env_sell_above, 
#                     avg_sim_buy_below, avg_sim_sell_above, avg_env_sell_above_best, avg_env_buy_below_est,
#                     avg_sim_sell_above_best, avg_sim_buy_below_est, nonAvg_value_agents, nonAvg_value_agents_non):
def pad_subarrays(arr1, arr2, pad_value=np.nan):
    # Find the length of the longest subarray
    max_len = max(len(arr1), len(arr2))
    
    # Pad each subarray to the maximum length
    padded_array = np.array([np.pad(subarray, (0, max_len - len(subarray)), constant_values=pad_value) for subarray in [arr1, arr2]])
    
    return padded_array

def loop_all_files(data_path, avg_env_best_buys, 
                    avg_env_best_asks, avg_env_spread, avg_env_trades):

    for dir in os.listdir(data_path):
        print(f"DIR: {dir}")
        if dir == "graphs":
            continue
        else:
            global PICKLE_PATH
            PICKLE_PATH = data_path + "/" + dir + "/pickle"
            env_best_buys = load_pickle(PICKLE_PATH + "/env_best_buys")
            env_best_asks = load_pickle(PICKLE_PATH + "/env_best_asks")
            env_trades = load_pickle(PICKLE_PATH + "/trades_env")
            avg_env_best_buys.append(np.nanmean(env_best_buys, axis=0))
            avg_env_best_asks.append(np.nanmean(env_best_asks, axis=0))
            avg_env_spread.append(np.nanmean(np.subtract(env_best_asks, env_best_buys), axis=0))
            avg_env_trades.append([np.unique(arr[~np.isnan(arr)]) for arr in env_trades])
            
    return avg_env_best_buys, avg_env_best_asks, avg_env_spread, avg_env_trades

def trim_list(listTrim):
    vals = []
    for val in listTrim:
        if len(vals) == 5:
            break
        if isinstance(val, np.ndarray) or isinstance(val, list):
            vals.append(val)
    return vals

def run():
    SIM_TIME = 10000
    NUM_AGENTS = 25
    paths = ["A1", "A2", "A3", "B1","B2", "B3", "C1", "C2", "C3"]
    for data_path in paths:
        print("data_path", data_path)
        tuned_path = BASE_PATH + data_path
        graph_path = tuned_path + "/graphs"
        os.makedirs(graph_path, exist_ok=True)
        
        avg_env_best_buys = list()
        avg_env_best_asks = list()
        avg_env_spread = list()
        avg_env_trades = list()

        loop_all_files(tuned_path, avg_env_best_buys, avg_env_best_asks, avg_env_spread, avg_env_trades)

        avg_env_best_buys = np.nanmean(avg_env_best_buys, axis=0)
        avg_env_best_asks = np.nanmean(avg_env_best_asks, axis=0)
        avg_env_spread = np.nanmean(avg_env_spread, axis=0)
        avg_num_trades = []
        for elem in avg_env_trades:
            curr_sim = []
            for sim in elem:
                curr_sim.append(len(sim))
            avg_num_trades.extend(curr_sim)
            
        f = open(graph_path + "/best_ask.txt", "w")
        print(list(avg_env_best_asks), file=f)
        f.close()

        f = open(graph_path + "/best_buy.txt", "w")
        print(list(avg_env_best_buys), file=f)
        f.close()

        f = open(graph_path + "/spread.txt", "w")
        print(list(avg_env_spread), file=f)
        f.close()

        f = open(graph_path + "/trades.txt", "w")
        print(np.nanmean(avg_num_trades, axis=0), file=f)
        f.close()

run()
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import gzip
from fourheap.constants import BUY, SELL


BASE_PATH = "tuned_buys_final/6e3_spoofer/"
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
def weighted_increase_sum(arr, est_funds, trades):
    total_sum = 0
    total_surplus_lost = 0
    for i in range(len(arr) - 1):
        if arr[i + 1] > arr[i]:
            total_sum += arr[i + 1] - arr[i]
            total_surplus_lost += (trades[i + 1] - est_funds[-2])
    return total_sum, total_surplus_lost

def loop_all_files(data_path, avg_spoof_position, avg_loss):

    full_data_path = BASE_PATH + data_path
    for root, dirs, files in os.walk(full_data_path):
        print(f"Root: {root}")
        dirs_exclude = []
        for dir in dirs:
            try:
                dirs_exclude.append(int(dir))
            except:
                print("NOT ADDED", dir)
                pass
        dirs_sorted_2 = [str(dir) for dir in sorted(dirs_exclude)]
        for dir in dirs_sorted_2:
                print(f"Subdirectory: {dir}")
                global PICKLE_PATH
                PICKLE_PATH = full_data_path + "/" + dir + "/pickle"
                spoofer_position = load_pickle(PICKLE_PATH + "/position_env")
                est_funds = load_pickle(PICKLE_PATH + "/env_est_funds")
                trades = load_pickle(PICKLE_PATH + "/trades_env")

                buys = []
                loss = []
                for ind, sim in enumerate(spoofer_position):
                    num_buys, surplus_lost = weighted_increase_sum(sim, est_funds[ind], trades[ind])
                    buys.append(num_buys)
                    loss.append(surplus_lost)

                avg_spoof_position.append(np.nanmean(buys))
                avg_loss.append(np.nanmean(loss))
                
        path = BASE_PATH + data_path + "/graphs"
        f = open(path + "/check_position_2.txt", "a")
        print(avg_spoof_position, file=f)
        print(avg_loss, file=f)
        f.close()
        
        f = open(path + "/check_position_3.txt", "w")
        print(buys, file=f)
        print(loss, file=f)
        f.close()

    # return avgValueAgents, avgValueNon, avg_env_trades, avg_sim_trades, avg_spoof_position, avg_non_position, avg_diffs, avg_spoofer_surplus, avg_nonspoofer_surplus, avg_env_est_fund, avg_env_sell_orders, avg_sim_sell_orders, avg_env_spoof_orders, avg_sim_spoof_orders, avg_env_best_buys, avg_sim_best_buys, avg_env_best_asks, avg_sim_best_asks, avg_env_buy_below, avg_env_sell_above, avg_sim_buy_below, avg_sim_sell_above, avg_env_sell_above_best, avg_env_buy_below_est, avg_sim_sell_above_best, avg_sim_buy_below_est, nonAvg_value_agents, nonAvg_value_agents_non
    return avg_spoof_position, avg_loss

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
    # paths = ["B2"]
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    for data_path in paths:
        path = BASE_PATH + data_path + "/graphs"
        os.makedirs(path, exist_ok=True)
        print("GRAPH SAVE PATH = ", path)

        avg_spoof_position = list() 
        avg_loss = list()
        
        avg_spoof_position, avg_loss = loop_all_files(data_path, avg_spoof_position, avg_loss)

        # avg_spoof_position = np.nanmean(trim_list(avg_spoof_position), axis=0)
        # avg_loss = np.nanmean(trim_list(avg_loss), axis=0)
        # f = open(path + "/check_position.txt", "a")
        # print(avg_spoof_position, file=f)
        # print(avg_loss, file=f)
        # f.close()
      
run()
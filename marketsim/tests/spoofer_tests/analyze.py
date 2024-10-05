'''
    Analyzing the data pickle files saved by the simulation and outputting more readable txt data files.
    These data files can then be parsed and graphed.
'''

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import gzip


BASE_PATH = "DEFINE BASE PATH HERE"
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

def loop_all_files(data_path, avgValueAgents, avg_env_trades, 
                    avg_spoof_position, avg_spoofer_surplus, 
                    avg_env_est_fund, avg_env_sell_orders, 
                    avg_env_spoof_orders, avg_env_best_buys, 
                    avg_env_best_asks, avg_env_buy_below, avg_env_sell_above, 
                    avg_env_sell_above_best, avg_env_buy_below_est,
                    nonAvg_value_agents):

    data_path = BASE_PATH + data_path
    for dir in os.listdir(data_path):
        print(f"Dir: {dir}")
        if dir == "graphs":
            continue
        else:
            print(f"Subdirectory: {dir}")
            global PICKLE_PATH
            PICKLE_PATH = data_path + "/" + dir + "/pickle"
            valueAgentsSpoof = load_pickle(PICKLE_PATH + "/values_env")
            env_trades = load_pickle(PICKLE_PATH + "/trades_env")
            spoofer_position = load_pickle(PICKLE_PATH + "/position_env")
            spoofer_surplus = load_pickle(PICKLE_PATH + "/surplus_env")
            env_est_fund = load_pickle(PICKLE_PATH + "/env_est_funds")
            env_sell_orders = load_pickle(PICKLE_PATH + "/env_sell_orders")
            env_spoof_orders = load_pickle(PICKLE_PATH + "/env_spoof_orders")
            env_best_buys = load_pickle(PICKLE_PATH + "/env_best_buys")
            env_best_asks = load_pickle(PICKLE_PATH + "/env_best_asks")
            nonAvg_value_agents.extend(valueAgentsSpoof)
            avgValueAgents.append(np.mean(valueAgentsSpoof, axis=0))
            avg_env_trades.append(np.nanmean(env_trades, axis = 0))
            avg_spoof_position.append(np.nanmean(spoofer_position, axis=0))
            avg_spoofer_surplus.append(np.nanmean(spoofer_surplus, axis=0))
            avg_env_est_fund.append(np.nanmean(env_est_fund, axis=0))
            avg_env_sell_orders.append(np.nanmean(env_sell_orders, axis=0))
            avg_env_spoof_orders.append(np.nanmean(env_spoof_orders, axis=0))
            avg_env_best_buys.append(np.nanmean(env_best_buys, axis=0))
            avg_env_best_asks.append(np.nanmean(env_best_asks, axis=0))
            avg_env_buy_below.append(np.nanmean(np.subtract(env_spoof_orders, env_best_buys), axis=0))
            avg_env_sell_above.append(np.nanmean(np.subtract(env_sell_orders, env_est_fund), axis=0))
            avg_env_sell_above_best.append(np.nanmean(np.subtract(env_sell_orders, env_best_asks), axis=0))
            avg_env_buy_below_est.append(np.nanmean(np.subtract(env_spoof_orders, env_est_fund), axis=0))

    return avgValueAgents,  avg_env_trades, avg_spoof_position, avg_spoofer_surplus, avg_env_est_fund, avg_env_sell_orders, avg_env_spoof_orders, avg_env_best_buys, avg_env_best_asks, avg_env_buy_below, avg_env_sell_above, avg_env_sell_above_best, avg_env_buy_below_est, nonAvg_value_agents
    
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
    paths = ["A1", "A2", "A3", "B1" ,"B2", "B3", "C1",  "C2", "C3"]
    for data_path in paths:
        path = BASE_PATH + data_path + "/graphs"
        os.makedirs(path, exist_ok=True)
        print("GRAPH SAVE PATH = ", path)

        avgValueAgents = list()
        avg_env_trades = list()
        avg_spoof_position = list() 
        avg_spoofer_surplus = list()
        avg_env_est_fund = list()
        avg_env_sell_orders = list()
        avg_env_spoof_orders = list()
        avg_env_best_buys = list()
        avg_env_best_asks = list()
        avg_env_buy_below = list()
        avg_env_sell_above = list()
        avg_env_sell_above_best = list()
        avg_env_buy_below_est = list()
        avg_env_sell_above_best = list()
        avg_env_buy_below_est = list()
        nonAvg_value_agents = list()
        
        loop_all_files(data_path, avgValueAgents, avg_env_trades, avg_spoof_position, avg_spoofer_surplus, avg_env_est_fund, avg_env_sell_orders, avg_env_spoof_orders, avg_env_best_buys, avg_env_best_asks, avg_env_buy_below, avg_env_sell_above, avg_env_sell_above_best, avg_env_buy_below_est, nonAvg_value_agents)

        newNonAvg = []
        for val in nonAvg_value_agents:
            if isinstance(val, np.ndarray) or isinstance(val, list):
                newNonAvg.append([np.mean(val[:12]), np.mean(val[12:24]), val[-2], val[-1]])
        avg = np.mean(newNonAvg, axis=0)
        std = np.std(newNonAvg, axis=0)

        f = open(path + "/avgstd.txt", "w")
        print(avg, file=f)
        print(std, file=f)
        f.close()


        avg_env_trades = np.nanmean(trim_list(avg_env_trades), axis=0)
        
        avg_spoof_position = np.nanmean(trim_list(avg_spoof_position), axis=0)
        avg_spoofer_surplus = np.nanmean(trim_list(avg_spoofer_surplus), axis=0)
        avg_env_est_fund = np.nanmean(trim_list(avg_env_est_fund), axis=0)
        avg_env_sell_orders = np.nanmean(trim_list(avg_env_sell_orders), axis=0)
        avg_env_spoof_orders = np.nanmean(trim_list(avg_env_spoof_orders), axis=0)
        avg_env_best_buys = np.nanmean(trim_list(avg_env_best_buys), axis=0)
        avg_env_best_asks = np.nanmean(trim_list(avg_env_best_asks), axis=0)
        avg_env_buy_below = np.nanmean(trim_list(avg_env_buy_below), axis=0)
        avg_env_sell_above = np.nanmean(trim_list(avg_env_sell_above), axis=0)
        avg_env_sell_above_best = np.nanmean(trim_list(avg_env_sell_above_best), axis=0)
        avg_env_buy_below_est = np.nanmean(trim_list(avg_env_buy_below_est), axis=0)

        x_axis = [i for i in range(0, SIM_TIME+1)]

        plt.figure()
        plt.plot(x_axis, avg_env_trades, label="MM")
        plt.legend()
        plt.xlabel('Timesteps')
        plt.ylabel('Transaction Price')
        plt.title('Last Transaction Prices for Different MM Configs')
        plt.savefig(path + '/AVG_matched_order_price.jpg')
        plt.close()

        plt.figure()
        plt.plot(x_axis, avg_spoof_position, label="MM")
        plt.xlabel('Timesteps')
        plt.ylabel('Position')
        plt.title('Averaged Spoofer Position')
        plt.legend()
        plt.savefig(path + '/AVG_spoofer_position.jpg')
        plt.close()

        plt.figure()
        plt.plot(x_axis, avg_spoofer_surplus, label="Env Surplus")
        plt.xlabel('Timesteps')
        plt.ylabel('Surplus')
        plt.title('Averaged Surplus of Spoofer')
        plt.savefig(path + '/AVG_spoofer_surplus_track.jpg')
        plt.close()

        plt.figure()
        plt.plot(x_axis, avg_env_spoof_orders, label="env spoof")
        plt.plot(x_axis, avg_env_best_buys, label="env best buys", linestyle="--")
        plt.plot(x_axis, avg_env_best_asks, label="env best asks", linestyle="--")
        plt.plot(x_axis, avg_env_sell_orders, label="env sell orders")

        plt.legend()
        plt.savefig(path + '/AVG_orders.jpg')
        plt.close()

        plt.figure()
        plt.plot(x_axis, np.add(avg_env_best_buys, avg_env_best_asks) / 2, label="env midprice")
        plt.legend()
        plt.savefig(path + '/AVG_midprice.jpg')
        plt.close()

        plt.figure()
        plt.plot(x_axis, avg_env_sell_above,label="env sells")
        plt.xlabel('Timestep')
        plt.ylabel('Spoof sell - estimated fundamental')
        plt.title('Sell order relative to estimated fundamental')
        plt.legend()
        plt.savefig(path + '/AVG_sell_above_est-fund.jpg')
        plt.close()

        plt.figure()
        plt.plot(x_axis, avg_env_buy_below, color = 'cyan', label="env buys")
        # plt.plot(x_axis, avg_sim_buy_below, color = "magenta", linestyle= "dotted", label="sim buys")
        plt.xlabel('Timesteps')
        plt.ylabel('Buy spoof - best buy')
        plt.title('Spoof Buy Orders Relative to Best Buy')
        plt.legend()
        plt.savefig(path + '/AVG_buy_below_best_ask.jpg')
        plt.close()

        plt.figure()
        bar_width = 0.35
        num_agents = [j for j in range(NUM_AGENTS + 1)]
        # num_agent_non= [x + bar_width for x in num_agents]
        plt.bar(num_agents, np.mean(trim_list(avgValueAgents), axis=0), color='b', width=bar_width, edgecolor='grey', label='MM')
        plt.legend()
        plt.title('Surplus Comparison')
        plt.xlabel('Agent')
        plt.ylabel('Values')
        plt.xticks([r + bar_width/2 for r in range(len(num_agents))], num_agents)
        plt.savefig(path + '/surpluses_sim.jpg')
        plt.close()

        f = open(path + "/values.txt", "w")
        print("Non BASE MM AVERAGE SURPLUS", file = f)
        print(np.mean(trim_list(avgValueAgents), axis=0), file = f)
        f.close()


        avgValueAgentsTrimmed = trim_list(avgValueAgents)
        newVals = []
        for valArr in avgValueAgentsTrimmed:
            newVal = [np.mean(valArr[:12]), np.mean(valArr[12:24]), valArr[-2], valArr[-1]]
            newVals.append(newVal)

        f = open(path + "/valuesCompare.txt", "w")
        print("ValueAgent", file = f)
        print(np.mean(newVals, axis=0), file=f)
        print(np.std(newVals, axis=0), file=f)
        f.close()
        
        f = open(path + "/midprice.txt", "w")
        print("AVERAGE MIDPRICE", file = f)
        print(list(np.add(avg_env_best_buys, avg_env_best_asks) / 2), file=f)
        f.close()

        f = open(path + "/position.txt", "w")
        print(list(avg_spoof_position), file=f)
        f.close()

        f = open(path + "/env_above_best.txt", "w")
        print("ENV SELL ABOVE BEST", file=f)
        print(list(avg_env_sell_above), file=f)
        f.close()

        f = open(path + "/env_below_best.txt", "w")
        print("ENV BUY BELOW", file=f)
        print(list(avg_env_buy_below), file=f)
        f.close()

        f = open(path + "/trades.txt", "w")
        # print("ENV BUY BELOW", file=f)
        print(list(avg_env_trades), file=f)
        f.close()
        
        f = open(path + "/env_sell_above_best.txt", "w")
        print(list(avg_env_sell_above_best), file=f)
        f.close()
        
        f = open(path + "/env_buy_below_est.txt", "w")
        print(list(avg_env_buy_below_est), file=f)
        f.close()

run()
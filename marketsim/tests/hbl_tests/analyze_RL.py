from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import gzip
from fourheap.constants import BUY, SELL


BASE_PATH = "official_rl_exps_2_action/2e2_spoofer/"
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
            # valuesAgentsNon = load_pickle(PICKLE_PATH + "/values_sim")
            env_trades = load_pickle(PICKLE_PATH + "/trades_env")
            # sim_trades = load_pickle(PICKLE_PATH + "/trades_sim")
            spoofer_position = load_pickle(PICKLE_PATH + "/position_env")
            # nonspoofer_position = load_pickle(PICKLE_PATH + "/position_sim")
            spoofer_surplus = load_pickle(PICKLE_PATH + "/surplus_env")
            # nonspoofer_surplus = load_pickle(PICKLE_PATH + "/surplus_sim")
            env_est_fund = load_pickle(PICKLE_PATH + "/env_est_funds")
            # sim_est_fund = load_pickle(PICKLE_PATH + "/sim_est_funds")
            env_sell_orders = load_pickle(PICKLE_PATH + "/env_sell_orders")
            # sim_sell_orders = load_pickle(PICKLE_PATH + "/sim_sell_orders")
            env_spoof_orders = load_pickle(PICKLE_PATH + "/env_spoof_orders")
            # sim_spoof_orders = load_pickle(PICKLE_PATH + "/sim_spoof_orders")
            env_best_buys = load_pickle(PICKLE_PATH + "/env_best_buys")
            # sim_best_buys = load_pickle(PICKLE_PATH + "/sim_best_buys")
            env_best_asks = load_pickle(PICKLE_PATH + "/env_best_asks")
            # sim_best_asks = load_pickle(PICKLE_PATH + "/sim_best_asks")
            nonAvg_value_agents.extend(valueAgentsSpoof)
            # nonAvg_value_agents_non.extend(valuesAgentsNon)
            avgValueAgents.append(np.mean(valueAgentsSpoof, axis=0))
            # avgValueNon.append(np.mean(valuesAgentsNon, axis=0))
            avg_env_trades.append(np.nanmean(env_trades, axis = 0))
            # avg_sim_trades.append(np.nanmean(sim_trades, axis = 0))
            # avg_diffs.append(np.subtract(np.nanmean(env_trades, axis = 0), np.nanmean(sim_trades, axis = 0)))
            avg_spoof_position.append(np.nanmean(spoofer_position, axis=0))
            # avg_non_position.append(np.nanmean(nonspoofer_position, axis=0))
            avg_spoofer_surplus.append(np.nanmean(spoofer_surplus, axis=0))
            # avg_nonspoofer_surplus.append(np.nanmean(nonspoofer_surplus,axis = 0))
            avg_env_est_fund.append(np.nanmean(env_est_fund, axis=0))
            avg_env_sell_orders.append(np.nanmean(env_sell_orders, axis=0))
            # avg_sim_sell_orders.append(np.nanmean(sim_sell_orders, axis=0))
            avg_env_spoof_orders.append(np.nanmean(env_spoof_orders, axis=0))
            # avg_sim_spoof_orders.append(np.nanmean(sim_spoof_orders, axis=0))
            avg_env_best_buys.append(np.nanmean(env_best_buys, axis=0))
            # avg_sim_best_buys.append(np.nanmean(sim_best_buys, axis=0))
            avg_env_best_asks.append(np.nanmean(env_best_asks, axis=0))
            # avg_sim_best_asks.append(np.nanmean(sim_best_asks, axis=0))
            avg_env_buy_below.append(np.nanmean(np.subtract(env_spoof_orders, env_best_buys), axis=0))
            avg_env_sell_above.append(np.nanmean(np.subtract(env_sell_orders, env_est_fund), axis=0))
            # avg_sim_buy_below.append(np.nanmean(np.subtract(sim_spoof_orders, sim_best_buys), axis=0))
            # avg_sim_sell_above.append(np.nanmean(np.subtract(sim_sell_orders, env_est_fund), axis=0))
            avg_env_sell_above_best.append(np.nanmean(np.subtract(env_sell_orders, env_best_asks), axis=0))
            avg_env_buy_below_est.append(np.nanmean(np.subtract(env_spoof_orders, env_est_fund), axis=0))
            # avg_sim_sell_above_best.append(np.nanmean(np.subtract(sim_sell_orders, sim_best_asks), axis=0))
            # avg_sim_buy_below_est.append(np.nanmean(np.subtract(sim_spoof_orders, env_est_fund), axis=0))
            path = BASE_PATH + data_path + "/rl_vals"
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=False)
            f = open(path + "/avgstd.txt", "a")
            mean = np.mean(valueAgentsSpoof, axis=0)
            std = np.std(valueAgentsSpoof, axis=0)
            print(mean)
            print(std)
            print("PATH", dir, file=f)
            print([np.mean(mean[:12]), np.mean(mean[12:24]), mean[24], mean[25]], file=f)
            print([np.std(mean[:12]), np.std(mean[12:24]), std[24], std[25]], file=f)
            print("\n\n")
            f.close()
    # return avgValueAgents, avgValueNon, avg_env_trades, avg_sim_trades, avg_spoof_position, avg_non_position, avg_diffs, avg_spoofer_surplus, avg_nonspoofer_surplus, avg_env_est_fund, avg_env_sell_orders, avg_sim_sell_orders, avg_env_spoof_orders, avg_sim_spoof_orders, avg_env_best_buys, avg_sim_best_buys, avg_env_best_asks, avg_sim_best_asks, avg_env_buy_below, avg_env_sell_above, avg_sim_buy_below, avg_sim_sell_above, avg_env_sell_above_best, avg_env_buy_below_est, avg_sim_sell_above_best, avg_sim_buy_below_est, nonAvg_value_agents, nonAvg_value_agents_non
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
    # paths = ["A2", "A3","B2", "B1", "C1", "C2", "C3"]
    # paths = ["A3_2", "C1_3"]
    # paths = ["A1", "B3"]
    # paths = ["130", "140", "150"]
    # paths = ["A1_5_8", "A3_80_90", "B2_55_60", "B3_110_120", "C2_80_90", "C3_140_145"]
    # paths = ["A2_35_40", "A3_70_80", "A3_90_100", "B1_45_50", "B2_55_65", "C1_75_80", "C2_130_135"]
    # paths = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    for data_path in paths:
        path = BASE_PATH + data_path + "/graphs"
        os.makedirs(path, exist_ok=True)
        print("GRAPH SAVE PATH = ", path)

        avgValueAgents = list()
        # avgValueNon = list()
        avg_env_trades = list()
        # avg_sim_trades = list()
        avg_spoof_position = list() 
        # avg_non_position = list()
        # avg_diffs = list()
        avg_spoofer_surplus = list()
        # avg_nonspoofer_surplus = list()
        avg_env_est_fund = list()
        avg_env_sell_orders = list()
        # avg_sim_sell_orders = list()
        avg_env_spoof_orders = list()
        # avg_sim_spoof_orders = list()
        avg_env_best_buys = list()
        # avg_sim_best_buys = list()
        avg_env_best_asks = list()
        # avg_sim_best_asks = list()
        avg_env_buy_below = list()
        avg_env_sell_above = list()
        avg_env_sell_above_best = list()
        avg_env_buy_below_est = list()
        # avg_sim_buy_below = list()
        # avg_sim_sell_above = list()
        # avg_sim_sell_above_best = list()
        # avg_sim_buy_below_est = list()
        avg_env_sell_above_best = list()
        avg_env_buy_below_est = list()
        # avg_sim_sell_above_best = list()
        # avg_sim_buy_below_est = list()
        nonAvg_value_agents = list()
        # nonAvg_value_agents_non = list()
        
        # loop_all_files(data_path, avgValueAgents, avgValueNon, avg_env_trades, avg_sim_trades, avg_spoof_position, avg_non_position, avg_diffs, avg_spoofer_surplus, avg_nonspoofer_surplus, avg_env_est_fund, avg_env_sell_orders, avg_sim_sell_orders, avg_env_spoof_orders, avg_sim_spoof_orders, avg_env_best_buys, avg_sim_best_buys, avg_env_best_asks, avg_sim_best_asks, avg_env_buy_below, avg_env_sell_above, avg_sim_buy_below,avg_sim_sell_above, avg_env_sell_above_best, avg_env_buy_below_est, avg_sim_sell_above_best, avg_sim_buy_below_est, nonAvg_value_agents, nonAvg_value_agents_non)
        loop_all_files(data_path, avgValueAgents, avg_env_trades, avg_spoof_position, avg_spoofer_surplus, avg_env_est_fund, avg_env_sell_orders, avg_env_spoof_orders, avg_env_best_buys, avg_env_best_asks, avg_env_buy_below, avg_env_sell_above, avg_env_sell_above_best, avg_env_buy_below_est, nonAvg_value_agents)

        newNonAvg = []
        for val in nonAvg_value_agents:
            if isinstance(val, np.ndarray) or isinstance(val, list):
                newNonAvg.append([np.mean(val[:12]), np.mean(val[12:24]), val[-2], val[-1]])
        avg = np.mean(newNonAvg, axis=0)
        std = np.std(newNonAvg, axis=0)

        # newNonAvg_non = []
        # for val in nonAvg_value_agents_non:
        #     if isinstance(val, np.ndarray) or isinstance(val, list):
        #         newNonAvg_non.append([np.mean(val[:12]), np.mean(val[12:24]), val[-2], val[-1]])
        # non_avg = np.mean(newNonAvg_non, axis=0)
        # non_std = np.std(newNonAvg_non, axis=0)

        f = open(path + "/avgstd.txt", "w")
        print(avg, file=f)
        print(std, file=f)
        f.close()

        # f = open(path + "/avgstd_non.txt", "w")
        # print(non_avg, file=f)
        # print(non_std, file=f)
        # f.close()

        avg_env_trades = np.nanmean(trim_list(avg_env_trades), axis=0)
        # avg_sim_trades = np.nanmean(trim_list(avg_sim_trades), axis=0)
        avg_spoof_position = np.nanmean(trim_list(avg_spoof_position), axis=0)
        # avg_non_position = np.nanmean(trim_list(avg_non_position), axis=0)
        # avg_diffs = np.nanmean(trim_list(avg_diffs), axis=0)
        avg_spoofer_surplus = np.nanmean(trim_list(avg_spoofer_surplus), axis=0)
        # avg_nonspoofer_surplus = np.nanmean(trim_list(avg_nonspoofer_surplus), axis=0)
        avg_env_est_fund = np.nanmean(trim_list(avg_env_est_fund), axis=0)
        avg_env_sell_orders = np.nanmean(trim_list(avg_env_sell_orders), axis=0)
        # avg_sim_sell_orders = np.nanmean(trim_list(avg_sim_sell_orders), axis=0)
        avg_env_spoof_orders = np.nanmean(trim_list(avg_env_spoof_orders), axis=0)
        # avg_sim_spoof_orders = np.nanmean(trim_list(avg_sim_spoof_orders), axis=0)
        avg_env_best_buys = np.nanmean(trim_list(avg_env_best_buys), axis=0)
        # avg_sim_best_buys = np.nanmean(trim_list(avg_sim_best_buys), axis=0)
        avg_env_best_asks = np.nanmean(trim_list(avg_env_best_asks), axis=0)
        # avg_sim_best_asks = np.nanmean(trim_list(avg_sim_best_asks), axis=0)
        avg_env_buy_below = np.nanmean(trim_list(avg_env_buy_below), axis=0)
        avg_env_sell_above = np.nanmean(trim_list(avg_env_sell_above), axis=0)
        # avg_sim_buy_below = np.nanmean(trim_list(avg_sim_buy_below), axis=0)
        # avg_sim_sell_above = np.nanmean(trim_list(avg_sim_sell_above), axis=0)
        avg_env_sell_above_best = np.nanmean(trim_list(avg_env_sell_above_best), axis=0)
        avg_env_buy_below_est = np.nanmean(trim_list(avg_env_buy_below_est), axis=0)
        # avg_sim_sell_above_best = np.nanmean(trim_list(avg_sim_sell_above_best), axis=0)
        # avg_sim_buy_below_est = np.nanmean(trim_list(avg_sim_buy_below_est), axis=0)

        x_axis = [i for i in range(0, SIM_TIME+1)]

        # try:
        #     plt.figure()
        #     plt.plot(x_axis, avg_diffs)
        #     plt.title('Transaction Price Differences')
        #     plt.xlabel('Timesteps')
        #     plt.ylabel('Transaction Price Difference')
        #     plt.savefig(path + '/matched_diff.jpg')
        #     plt.close()
        # except:
        #     print("Error in diffs")

        try:
            plt.figure()
            plt.plot(x_axis, avg_env_trades, label="MM")
            # plt.plot(x_axis, avg_sim_trades, linestyle='dotted',label="Non MM")
            plt.legend()
            plt.xlabel('Timesteps')
            plt.ylabel('Transaction Price')
            plt.title('Last Transaction Prices for Different MM Configs')
            plt.savefig(path + '/AVG_matched_order_price.jpg')
            plt.close()
        except:
            print("Error in matched")

        try:
            plt.figure()
            plt.plot(x_axis, avg_spoof_position, label="MM")
            # plt.plot(x_axis, avg_non_position, linestyle="dotted", label="Non MM")
            plt.xlabel('Timesteps')
            plt.ylabel('Position')
            plt.title('Averaged Spoofer Position')
            plt.legend()
            plt.savefig(path + '/AVG_spoofer_position.jpg')
            plt.close()
        except:
            print("error in position")

        try:
            plt.figure()
            plt.plot(x_axis, avg_spoofer_surplus, label="Env Surplus")
            # plt.plot(x_axis, avg_nonspoofer_surplus, label="Sim Surplus")
            plt.xlabel('Timesteps')
            plt.ylabel('Surplus')
            plt.title('Averaged Surplus of Spoofer')
            plt.savefig(path + '/AVG_spoofer_surplus_track.jpg')
            plt.close()
        except:
            print("spoofer surplus")

        try:
            plt.figure()
            plt.plot(x_axis, avg_env_spoof_orders, label="env spoof")
            plt.plot(x_axis, avg_env_best_buys, label="env best buys", linestyle="--")
            plt.plot(x_axis, avg_env_best_asks, label="env best asks", linestyle="--")
            plt.plot(x_axis, avg_env_sell_orders, label="env sell orders")
            # plt.plot(x_axis, avg_sim_spoof_orders, label="sim spoof", zorder=10)
            # plt.plot(x_axis, avg_sim_best_buys, label="sim best buys", linestyle="--")
            # plt.plot(x_axis, avg_sim_best_asks, label="sim best asks", linestyle="--")
            # plt.plot(x_axis, avg_sim_sell_orders, label="sim sell orders")
            plt.legend()
            plt.savefig(path + '/AVG_orders.jpg')
            plt.close()
        except:
            print("orders")

        try:
            plt.figure()
            plt.plot(x_axis, np.add(avg_env_best_buys, avg_env_best_asks) / 2, label="env midprice")
            plt.legend()
            plt.savefig(path + '/AVG_midprice.jpg')
            plt.close()
        except:
            print("midprice")

        try:
            plt.figure()
            plt.plot(x_axis, avg_env_sell_above,label="env sells")
            # plt.plot(x_axis, avg_sim_sell_above, linestyle="dotted", label="sim sells")
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
        except:
            print("buy below, sell below")

        try:
            plt.figure()
            bar_width = 0.35
            num_agents = [j for j in range(NUM_AGENTS + 1)]
            num_agent_non= [x + bar_width for x in num_agents]
            plt.bar(num_agents, np.mean(trim_list(avgValueAgents), axis=0), color='b', width=bar_width, edgecolor='grey', label='MM')
            # plt.bar(num_agent_non, np.mean(trim_list(avgValueNon), axis=0), color='g', width=bar_width, edgecolor='grey', label='Non MM')
            plt.legend()
            plt.title('Surplus Comparison')
            plt.xlabel('Agent')
            plt.ylabel('Values')
            plt.xticks([r + bar_width/2 for r in range(len(num_agents))], num_agents)
            plt.savefig(path + '/surpluses_sim.jpg')
            plt.close()
        except:
            print("values")


        f = open(path + "/values.txt", "w")
        print("Non BASE MM AVERAGE SURPLUS", file = f)
        print(np.mean(trim_list(avgValueAgents), axis=0), file = f)
        # print("BASELINE AVERAGE SURPLUS", file = f)
        # print(np.mean(trim_list(avgValueNon), axis=0), file = f)
        f.close()


        avgValueAgentsTrimmed = trim_list(avgValueAgents)
        newVals = []
        for valArr in avgValueAgentsTrimmed:
            newVal = [np.mean(valArr[:12]), np.mean(valArr[12:24]), valArr[-2], valArr[-1]]
            newVals.append(newVal)

        # avgValueNonTrimmed = trim_list(avgValueNon)
        # newValsNon = []
        # for valArr in avgValueNonTrimmed:
        #     newVal = [np.mean(valArr[:12]), np.mean(valArr[12:24]), valArr[-2], valArr[-1]]
        #     newValsNon.append(newVal)

        f = open(path + "/valuesCompare.txt", "w")
        print("ValueAgent", file = f)
        print(np.mean(newVals, axis=0), file=f)
        print(np.std(newVals, axis=0), file=f)
        # print("ValueNon", file=f)
        # print(np.mean(newValsNon, axis= 0), file=f)
        # print(np.std(newValsNon, axis = 0), file=f)
        f.close()
        
        f = open(path + "/midprice.txt", "w")
        print("AVERAGE MIDPRICE", file = f)
        print(list(np.add(avg_env_best_buys, avg_env_best_asks) / 2), file=f)
        f.close()

        f = open(path + "/position.txt", "w")
        print(list(avg_spoof_position), file=f)
        f.close()

        # f = open(path + "/static_position.txt", "w")
        # print(list(avg_non_position), file=f)
        # f.close()

        # f = open(path + "/sim_midprice.txt", "w")
        # print("AVERAGE MIDPRICE", file = f)
        # print(list(np.add(avg_sim_best_buys, avg_sim_best_asks) / 2), file=f)
        # f.close()

        f = open(path + "/env_above_best.txt", "w")
        print("ENV SELL ABOVE BEST", file=f)
        print(list(avg_env_sell_above), file=f)
        f.close()
        # f = open(path + "/sim_above_best.txt", "w")
        # print("SIM SELL ABOVE BEST", file=f)
        # print(list(avg_sim_sell_above), file=f)
        # f.close()
        f = open(path + "/env_below_best.txt", "w")
        print("ENV BUY BELOW", file=f)
        print(list(avg_env_buy_below), file=f)
        f.close()
        # f = open(path + "/sim_below_best.txt", "w")
        # print("SIM BUY BELOW", file=f)
        # print(list(avg_sim_buy_below), file=f)
        # f.close()

        # f = open(path + "/sim_below_best.txt", "w")
        # print("TRANSACTION PRICE DIFF", file=f)
        # print(list(avg_diffs), file=f)
        # f.close()
        
        f = open(path + "/env_sell_above_best.txt", "w")
        print(list(avg_env_sell_above_best), file=f)
        f.close()
        
        # f = open(path + "/sim_sell_above_best.txt", "w")
        # print(list(avg_sim_sell_above_best), file=f)
        # f.close()
        
        f = open(path + "/env_buy_below_est.txt", "w")
        print(list(avg_env_buy_below_est), file=f)
        f.close()
        
        # f = open(path + "/sim_buy_below_est.txt", "w")
        # print(list(avg_sim_buy_below_est), file=f)
        # f.close()

run()
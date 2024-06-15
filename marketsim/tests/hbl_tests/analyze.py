from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from fourheap.constants import BUY, SELL
from spoof_mm_test import BASE_PATH, PICKLE_PATH

def load_pickle(file_path):
    """
    Load all objects from a pickle file.
    """
    file_path += ".pkl"
    data = None
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            while True:
                try:
                    if data == None:
                        data = pickle.load(f)
                    else:
                        data.extend(pickle.load(f))
                except EOFError:
                    break
    return data

SIM_TIME = 10000
NUM_AGENTS = 25
path = BASE_PATH + "/graphs"
os.makedirs(path, exist_ok=True)

print("GRAPH SAVE PATH = ", path)
   
valueAgentsSpoof = load_pickle(PICKLE_PATH + "/values_env")
valuesAgentsNon = load_pickle(PICKLE_PATH + "/values_sim")
env_trades = load_pickle(PICKLE_PATH + "/trades_env")
sim_trades = load_pickle(PICKLE_PATH + "/trades_sim")
spoofer_position = load_pickle(PICKLE_PATH + "/position_env")
nonspoofer_position = load_pickle(PICKLE_PATH + "/position_sim")
spoofer_surplus = load_pickle(PICKLE_PATH + "/surplus_env")
nonspoofer_surplus = load_pickle(PICKLE_PATH + "/surplus_sim")
env_est_fund = load_pickle(PICKLE_PATH + "/env_est_funds")
sim_est_fund = load_pickle(PICKLE_PATH + "/sim_est_funds")
env_sell_orders = load_pickle(PICKLE_PATH + "/env_sell_orders")
sim_sell_orders = load_pickle(PICKLE_PATH + "/sim_sell_orders")
env_spoof_orders = load_pickle(PICKLE_PATH + "/env_spoof_orders")
sim_spoof_orders = load_pickle(PICKLE_PATH + "/sim_spoof_orders")
env_best_buys = load_pickle(PICKLE_PATH + "/env_best_buys")
sim_best_buys = load_pickle(PICKLE_PATH + "/sim_best_buys")
env_best_asks = load_pickle(PICKLE_PATH + "/env_best_asks")
sim_best_asks = load_pickle(PICKLE_PATH + "/sim_best_asks")


x_axis = [i for i in range(0, SIM_TIME+1)]

plt.figure()
avg_env_trades = np.nanmean(env_trades, axis = 0)
avg_sim_trades = np.nanmean(sim_trades, axis = 0)

plt.plot(x_axis, np.subtract(avg_env_trades, avg_sim_trades))
plt.title('Transaction Price Differences')
plt.xlabel('Timesteps')
plt.ylabel('Transaction Price Difference')
plt.savefig(path + '/matched_diff.png')
plt.close()

plt.figure()
plt.plot(x_axis, avg_env_trades, label="spoof")
plt.plot(x_axis, avg_sim_trades, linestyle='dotted',label="Nonspoof")
plt.legend()
plt.xlabel('Timesteps')
plt.ylabel('Transaction Price')
plt.title('Last Transaction Prices for Different MM Configs')
plt.savefig(path + '/AVG_matched_order_price.png')
plt.close()

# plt.figure()
# plt.plot(x_axis, list(env.most_recent_trade.values()), label="spoof",  color="green")
# if PAIRED:
#     plt.plot(x_axis, list(sim.most_recent_trade.values()), linestyle='--',label="Nonspoof", color="orange")
# plt.legend()
# plt.xlabel('Timesteps')
# plt.ylabel('Last matched order price')
# plt.title('Spoof v Nonspoof last matched trade price - NOT AVERAGED')
# plt.savefig(path + '/{}_NONAVG_matched_order_price.png'.format(i))
# plt.close()

plt.figure()
plt.plot(x_axis, np.nanmean(spoofer_position, axis=0), label="spoof")
plt.plot(x_axis, np.nanmean(nonspoofer_position, axis=0), linestyle="dotted", label="nonspoof")
plt.xlabel('Timesteps')
plt.ylabel('Position')
plt.title('Averaged Spoofer Position')
plt.legend()
plt.savefig(path + '/AVG_spoofer_position.png')
plt.close()

plt.figure()
plt.plot(x_axis, np.nanmean(spoofer_surplus, axis=0), label="Env Surplus")
plt.plot(x_axis, np.nanmean(spoofer_surplus, axis=0), label="Sim Surplus")
plt.xlabel('Timesteps')
plt.ylabel('Surplus')
plt.title('Averaged Surplus of Spoofer')
plt.savefig(path + '/AVG_spoofer_surplus_track.png')
plt.close()

# plt.figure()
# plt.plot(x_axis, list(env.spoof_activity.values()), label="Surplus")
# plt.xlabel('Timesteps')
# plt.ylabel('Surplus')
# plt.title('Not AVG - Surplus of Spoofer Over Time')
# plt.savefig(path + '/{}_NONAVG_spoofer_surplus_track.png'.format(i))
# plt.close()

# plt.figure()
# plt.plot(x_axis, list(env.spoofer_quantity.values()), label="Position")
# plt.xlabel('Timesteps')
# plt.ylabel('Position')
# plt.title('NOTAVERAGED - Quantity of Spoofer Over Time')
# plt.savefig(path + '/{}_NONAVG_spoofer_position.png'.format(i))
# plt.close()

# plt.figure()
# plt.plot(x_axis, np.nanmean(spoof_mid_prices, axis=0), label="Spoof")
# plt.plot(x_axis, np.nanmean(nonspoof_mid_prices, axis=0), label="Nonspoof")
# plt.xlabel('Timesteps')
# plt.ylabel('Midprice')
# plt.legend()
# plt.title('AVERAGED - Midprice Spoof v Nonspoof')
# plt.savefig(path + '/{}_AVG_midprice.png'.format(i))
# plt.close()

# plt.figure()
# plt.hist(range(len(list(env.trade_volume.values()))), bins=len(list(env.trade_volume.values()))//100, weights=list(env.trade_volume.values()), edgecolor='black')
# plt.xlabel('Timesteps')
# plt.ylabel('# trades')
# plt.title('Spoof trade volume')
# plt.savefig(path + '/{}_NONAVG_trade_volume_spoof.png'.format(i))
# plt.close()

# if PAIRED:
#     plt.figure()
#     plt.hist(range(len(list(sim.trade_volume.values()))), bins=len(list(sim.trade_volume.values()))//100, weights=list(sim.trade_volume.values()), edgecolor='black')
#     plt.xlabel('Timesteps')
#     plt.ylabel('# trades')
#     plt.title('Nonspoof trade volume')
#     plt.savefig(path + '/{}_NONAVG_trade_volume_nonspoof.png'.format(i))
#     plt.close()

plt.figure()
plt.scatter(x_axis, env_spoof_orders[-1], label="spoof", color="magenta", zorder=10, s=2)
plt.plot(x_axis, env_best_buys[-1], label="best buys", linestyle="--", color="cyan")
plt.plot(x_axis, env_best_asks[-1], label="best asks", linestyle="--", color="yellow")
plt.scatter(x_axis, env_sell_orders[-1], label="sell orders", color="black", s=2)
plt.legend()
plt.xlabel('Timesteps')
plt.ylabel('Price')
plt.title('Price comparisons of spoofer orders - NOT AVERAGED')
plt.savefig(path + '/{}_spoofer_orders.png'.format(SIM_TIME))
plt.close()

avg_sell_diff_env = np.nanmean(np.subtract(np.array(env_sell_orders), np.array(env_est_fund)), axis=0)
avg_sell_diff_sim = np.nanmean(np.subtract(np.array(sim_sell_orders), np.array(sim_est_fund)), axis=0)
print(avg_sell_diff_sim[1000:2000])
plt.figure()
plt.plot(x_axis, avg_sell_diff_env, color = "cyan", label="env sells")
plt.plot(x_axis, avg_sell_diff_sim, color="magenta", linestyle="dotted", label="sim sells")
plt.xlabel('Timestep')
plt.ylabel('Spoof sell - best ask')
plt.title('Sell order relative to estimated fundamental')
plt.legend()
plt.savefig(path + '/AVG_sell_above_est-fund.png'.format(SIM_TIME))
plt.close()

avg_buy_diff_env = np.nanmean(np.subtract(np.array(env_spoof_orders), np.array(env_best_buys)), axis=0)
avg_buy_diff_sim = np.nanmean(np.subtract(np.array(sim_spoof_orders), np.array(sim_best_buys)), axis=0)
plt.figure()
plt.plot(x_axis, avg_buy_diff_env, color = 'cyan', label="env sells")
plt.plot(x_axis, avg_buy_diff_sim, color = "magenta", linestyle= "dotted", label="sim sells")
plt.xlabel('Timesteps')
plt.ylabel('Buy spoof - best buy')
plt.title('Spoof Buy Orders Relative to Best Buy')
plt.legend()
plt.savefig(path + '/AVG_buy_below_best_ask.png')
plt.close()


plt.figure()
bar_width = 0.35
num_agents = [j for j in range(NUM_AGENTS + 1)]
num_agent_non= [x + bar_width for x in num_agents]
plotSpoof = np.nanmean(valueAgentsSpoof, axis = 0)
plotNon = np.nanmean(valuesAgentsNon, axis = 0)
plt.bar(num_agent_non, plotNon, color='g', width=bar_width, edgecolor='grey', label='Nonspoof')
plt.bar(num_agents, plotSpoof, color='b', width=bar_width, edgecolor='grey', label='Spoof')
plt.legend()
plt.title('Surplus Comparison')
plt.xlabel('Agent')
plt.ylabel('Values')
plt.xticks([r + bar_width/2 for r in range(len(num_agents))], num_agents)
plt.savefig(path + '/surpluses_sim.png')
plt.close()

print("AVERAGE SURPLUS ENV", plotSpoof)
print("AVERAGE SURPLUS SIM", plotNon)
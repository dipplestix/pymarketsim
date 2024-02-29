import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from collections import defaultdict
from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.mean_reverting import GaussianMeanReverting
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.agent.hbl_agent import HBLAgent
from marketsim.simulator.simulator import Simulator
from marketsim.simulator.sampled_arrival_simulator import SimulatorSampledArrival
from marketsim.agent.zero_intelligence_agent import ZIAgent

NUM_AGENTS = 100
MEAN = 1e7
LAM = 1e-4
SIM_TIME = 6000
R = 0.05
SHOCK_VAR = 1e6
agents = {}

def graph_num_matched(num_matched):
    average_matched = []
    for lam_config in num_matched:
        average_matched.append(np.mean(num_matched[lam_config]))
    fig, ax = plt.subplots()
    ax.set_title("Number of matched orders")
    ax.plot(list(num_matched.keys()), average_matched)
    plt.show()

def graph_order_book_info(lam, buy_book, sell_book, buy_ask_spreads: List[tuple], fundamental, spread):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    #fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Order book size Lam = {}".format(lam))
    ax1.plot([i for i in range(len(buy_book))], buy_book, label="Buy")
    ax1.plot([i for i in range(len(sell_book))], sell_book, label="Sell")
    buy_ask_spreads = np.array(buy_ask_spreads)
    index = [i for i in range(len(buy_ask_spreads))]
    ax2.set_title("Order book spread (Max - Min)")
    ax2.plot(index, buy_ask_spreads[:, 0], label="Buy")
    ax2.plot(index, buy_ask_spreads[:, 1], label="Sell")
    ax2.legend()
    ax3.set_title("Fundamental values over time")
    ax3.plot(index, fundamental)
    ax3.set_xlabel('Index of individual agent entrance')
    # ax4.plot(index, spread)
    # ax4.set_xlabel("Spread between buy and ask books")
    fig.tight_layout()
    #fig2.tight_layout()
    plt.show()

def graph_matched_order_times(lam, matched_order_times):
    fig,ax = plt.subplots()
    y = [0] * len(matched_order_times["buy"])
    ax.set_title("Time it takes for buy/sell orders to match. Lam = {}".format(lam))
    ax.scatter(matched_order_times["buy"], y, color='blue', label='Buy', marker='o')  # Circle markers for type 1
    ax.scatter(matched_order_times["sell"], y, color='red', label='Sell', marker='^') 
    ax.set_xlabel("Time gap")
    ax.legend()
    plt.show()

#First test - varying lambdas
def varyLambdas():
    lambdas = [10 ** i for i in range(-5,0)]
    num_iters = 3
    num_matched_orders = defaultdict(list) 
    for lam in lambdas:
        total_order_book_buy = []
        total_order_book_sell = []
        total_buy_sell_spreads = []
        total_spreads = []
        total_fundamental_evol = []
        matched_orders = {"buy": [], "sell":[]}
        for _ in range(num_iters):
            fundamental = LazyGaussianMeanReverting(mean=MEAN, final_time=SIM_TIME, r=R, shock_var=SHOCK_VAR)
            markets = [Market(fundamental=fundamental, time_steps=SIM_TIME)]
            for i in range(NUM_AGENTS):
                agents[i] = ZIAgent(
                                agent_id=i,
                                market=markets[0],
                                q_max=20,
                                offset=12,
                                shade=[10, 30]
                            )
            sim = SimulatorSampledArrival(num_agents=NUM_AGENTS, sim_time=SIM_TIME, lam=lam, mean=MEAN, r=R, shock_var=SHOCK_VAR, agents=agents, markets=markets)
            sim.run()
            num_matched_orders[lam].append(len(sim.markets[0].matched_orders) / 2)
            total_order_book_buy.append(sim.ob_summary["buy"])
            total_order_book_sell.append(sim.ob_summary["sell"])
            for matched_order in sim.markets[0].matched_orders:
                if matched_order.order.order_type == BUY:
                    matched_orders["buy"].append(matched_order.time - matched_order.order.time)
                else:
                    matched_orders["sell"].append(matched_order.time - matched_order.order.time)
            total_buy_sell_spreads.append(sim.ob_max_spread)
            total_fundamental_evol.append(sim.fundamental_evol)
            total_spreads.append(sim.spread)

        avg_ob_summary_buy = get_average_1d_array(total_order_book_buy)
        avg_ob_summary_sell = get_average_1d_array(total_order_book_sell)
        avg_buy_ask_spreads = get_average_spread(total_buy_sell_spreads)
        avg_fundamental_evol = get_average_1d_array(total_fundamental_evol)
        avg_spreads = get_average_1d_array(total_spreads, contains_inf=True)
        graph_order_book_info(lam, avg_ob_summary_buy, avg_ob_summary_sell, avg_buy_ask_spreads, avg_fundamental_evol, avg_spreads)
        graph_matched_order_times(lam, matched_orders)
    print(num_matched_orders)
    graph_num_matched(num_matched_orders)
    return num_matched_orders

def get_average_1d_array(total_arr: List[List[any]], contains_inf = False):
    max_size = max(len(nested_list_values) for nested_list_values in total_arr)
    for nested_list in total_arr:
        if contains_inf:
            for i in range(len(nested_list)):
                if nested_list[i] == math.inf or nested_list[i] == -math.inf:
                    nested_list[i] = 0
        nested_list += [np.nan] * (max_size - len(nested_list))
    return np.nanmean(total_arr, axis=0)

def get_average_spread(total_spreads):
    max_size = max(len(order_spread) for order_spread in total_spreads)
    for order_summary in total_spreads:
        order_summary += [tuple([np.nan, np.nan])] * (max_size - len(order_summary))
    return np.nanmean(total_spreads, axis=0)


varyLambdas()





'''
    Graph bid ask spread of the market configurations.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
# from colors import COLORS
from colors import SINGLE_COLOR


OVERALL_PATHS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
def run():
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # paths = ["A1"]
    avg_spread = []
    avg_spread_std = []
    avg_num_trades = []
    for path in paths:
        with open(os.path.join(os.getcwd(), 'no_spoof','{}/graphs'.format(path), 'spread.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()
            for val in data:
                if val == "nan":
                    a.append(np.nan)
                else:
                    a.append(float(val))
            avg_spread.append(np.nanmean(a))
            avg_spread_std.append(np.nanstd(a))
        with open(os.path.join(os.getcwd(), 'no_spoof','{}/graphs'.format(path), 'trades.txt'), 'r') as file:
            data = file.read()
            avg_num_trades.append(float(val))


    plt.figure()
    plt.rcParams.update({'figure.autolayout': False})
    print(avg_spread_std)
    for ind,i in enumerate(avg_spread):
        plt.bar(ind, i, yerr=avg_spread_std[ind]/np.sqrt(10000), capsize=1, error_kw={'lw':1}, color=SINGLE_COLOR[0])

    plt.xticks(ticks=np.linspace(0,len(avg_spread), 9,endpoint=False), labels=paths, fontsize=18)
    plt.xlabel("Market Configuration", fontsize=20)
    plt.ylabel("Average Bid-Ask Spread", fontsize=20)
    plt.tick_params(axis='y', labelsize=18, labelcolor='black', width=2)
    plt.tight_layout()
    #TODO: Change path accordingly
    plt.savefig(os.getcwd() + "/[DATA_SAVE_PATH]" + "/spread.pdf")

run()
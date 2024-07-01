import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import savgol_filter

OVERALL_PATHS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
OVERALL_PATHS = ["A1"]
def run(valPath):
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    paths = ["A1"]
    env_above_best = []
   
    for path in paths:
        with open(os.path.join(os.getcwd(), 'baseline_no_MM/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'env_sell_above_best.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[:-1]
            for val in data:
                a.append(float(val))
            env_above_best.append(a)

    plt.figure(constrained_layout=True)
    x_axis = np.linspace(1,10000,10000)
    for ind,i in enumerate(env_above_best):
        # plt.plot(x_axis, i, label=paths[ind])
        y = savgol_filter(i, 50, 3)
        plt.plot(x_axis, y, zorder=100, label=OVERALL_PATHS[ind])

    plt.xlabel("Timesteps")
    plt.ticklabel_format(useOffset=False)
    # plt.ticklabel_format(style='plain')
    plt.ylabel("Order Price - Best Ask")
    plt.legend(ncols=3)
    plt.savefig(os.getcwd() + "/new_data_compil/bl_no_mm/{}_spoofer".format(valPath) + "/env_above_est.png")

def run_below(valPath):
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    paths = ["A1"]

    env_below_best = []
   
    for path in paths:
        with open(os.path.join(os.getcwd(), 'baseline_no_MM/{}_spoofer'.format(valPath), '{}/graphs'.format(path), "env_buy_below_est.txt"), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[:-1]
            for val in data:
                if val == "nan":
                    a.append(np.nan)
                else:
                    a.append(float(val))
            env_below_best.append(a)

    plt.figure(constrained_layout=True)
    x_axis = np.linspace(1,10000,10000)
    for ind,i in enumerate(env_below_best):
        # plt.plot(x_axis, i, label=paths[ind])
        y = savgol_filter(i, 100, 3)
        plt.plot(x_axis, y, zorder=100, label=OVERALL_PATHS[ind])

    plt.xlabel("Timesteps")
    plt.ticklabel_format(useOffset=False)
    # plt.ticklabel_format(style='plain')
    plt.ylabel("Order Price - Estimated Fundamental")
    plt.legend(ncols=3)
    plt.savefig(os.getcwd() + "/new_data_compil/bl_no_mm/{}_spoofer".format(valPath) + "/env_below_est.png")

valsPath = ["2e2", "6e3"]
valsPath = ["2e2"]
for valPath in valsPath:
    run(valPath)
    run_below(valPath)
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import savgol_filter

OVERALL_PATHS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

def run():
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "D1", "D2", "D3"]
    
    env_above_best = []
    for path in paths:
        with open(os.path.join(os.getcwd(), 'official_xw_benchmark/2e2_spoofer/', '{}/graphs'.format(path), 'env_above_best.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[4:]
            data = data[:-1]
            for val in data:
                if val == "nan":
                    a.append(np.nan)
                else:
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
    plt.ylabel("Order Price - Estimated Fundamental")
    plt.legend(ncols=3)
    plt.savefig(os.getcwd() + "/new_data_compil/2e2_spoofer" + "/rounded_env_above_best.png")

def run_below():
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "D1", "D2", "D3"]

    env_below_best = []
    for path in paths:
        with open(os.path.join(os.getcwd(), 'official_xw_benchmark/2e2_spoofer/', '{}/graphs'.format(path), "env_below_best.txt"), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[3:]
            data = data[:-1]
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
    plt.ylabel("Order Price - Best Buy")
    plt.legend(ncols=3)
    plt.savefig(os.getcwd() + "/new_data_compil/2e2_spoofer" + "/rounded_env_below_best.png")

def run_midprices():
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "D1", "D2", "D3"]    
    midprices = []

    for path in paths:
        with open(os.path.join(os.getcwd(), 'official_xw_benchmark/2e2_spoofer/', '{}/graphs'.format(path), 'midprice.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[2:]
            data = data[:-1]
            for val in data:
                a.append(float(val))
            midprices.append(a)

    plt.figure(constrained_layout=True)
    x_axis = np.linspace(1,10000,10000)
    for ind,i in enumerate(midprices):
        # plt.plot(x_axis, i, label=paths[ind])
        y = savgol_filter(i, 50, 3)
        plt.plot(x_axis, y, zorder=100, label=OVERALL_PATHS[ind])

    plt.xlabel("Timesteps")
    plt.ticklabel_format(useOffset=False)
    # plt.ticklabel_format(style='plain')
    plt.ylabel("Price")
    plt.legend(ncol=3)
    plt.savefig(os.getcwd() + "/new_data_compil/2e2_spoofer" + "/midprice.png")



run()
run_below()
run_midprices()
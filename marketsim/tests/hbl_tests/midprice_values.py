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
    midprices = []
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'baseline_no_MM/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'midprice.txt'), 'r') as file:
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
        plt.plot(x_axis, i, zorder=100, label=OVERALL_PATHS[ind])

    plt.xlabel("Timesteps")
    plt.ticklabel_format(useOffset=False)
    # plt.ticklabel_format(style='plain')
    plt.ylabel("Price")
    plt.legend(ncol=3)
    plt.savefig(os.getcwd() + "/new_data_compil/bl_no_mm/{}_spoofer".format(valPath) + "/midprice.png")

    
    midprices = []
    for path in paths:
        with open(os.path.join(os.getcwd(), 'baseline_no_MM/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'sim_midprice.txt'), 'r') as file:
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
        plt.plot(x_axis, i, zorder=100, label=OVERALL_PATHS[ind])

    plt.xlabel("Timesteps")
    plt.ticklabel_format(useOffset=False)
    # plt.ticklabel_format(style='plain')
    plt.ylabel("Price")
    plt.legend(ncol=3)
    plt.savefig(os.getcwd() + "/new_data_compil/bl_no_mm/{}_spoofer".format(valPath) + "/sim_midprice.png")



valsPath = ["2e2", "6e3"]
valsPath = ["2e2"]
for valPath in valsPath:
    run(valPath)
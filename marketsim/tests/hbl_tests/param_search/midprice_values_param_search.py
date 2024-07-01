import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import savgol_filter

OVERALL_PATHS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

def run(valPath, indPath):
    pathToTake = [{"A1":"s", "A2":"s", "A3":"s", "B1":"s", "B2":"s", "B3":"s", "C1":"s", "C2":"s", "C3":"s"},
{"A1":"n", "A2":"s", "A3":"s", "B1":"s", "B2":"s", "B3":"n", "C1":"s", "C2":"s", "C3":"n"}
    ]
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

    midprices = []
    static_midprices = []
    for path in paths:
        with open(os.path.join(os.getcwd(), 'param_search/{}_optimal'.format(valPath), '{}/graphs'.format(path), 'midprice.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[2:]
            data = data[:-1]
            for val in data:
                a.append(float(val))
            midprices.append(a)

        with open(os.path.join(os.getcwd(), 'param_search/{}_optimal'.format(valPath), '{}/graphs'.format(path), 'sim_midprice.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[2:]
            data = data[:-1]
            for val in data:
                a.append(float(val))
            static_midprices.append(a)

    plt.figure(constrained_layout=True)
    x_axis = np.linspace(1,10000,10000)
    for ind,i in enumerate(midprices):
        if pathToTake[indPath][paths[ind]] == "s":
            plot = static_midprices[ind]
        else:
            plot = i
        plt.plot(x_axis, plot, zorder=100, label=OVERALL_PATHS[ind])

    plt.xlabel("Timesteps")
    plt.ticklabel_format(useOffset=False)
    # plt.ticklabel_format(style='plain')
    plt.ylabel("Price")
    plt.legend(ncol=3)
    plt.savefig(os.getcwd() + "/new_data_compil/param_search/{}_spoofer".format(valPath) + "/noisy_midprice.png")



valsPath = ["2e2", "6e3"]
# valsPath = ["2e2", "6e3"]
for indPath,valPath in enumerate(valsPath):
    run(valPath, indPath)
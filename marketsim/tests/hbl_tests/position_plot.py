import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
from colors import COLORS
from matplotlib.ticker import ScalarFormatter

# valsPath = ["2e2"]
valsPath = ["2e2"]
# valsPath = ["6e3"]
for valPath in valsPath:
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # paths = ["A1"]
    pathNames = ["A1", "A2","A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # pathNames = ["A1"]
    values = []
   
    for path in paths:
        with open(os.path.join(os.getcwd(), 'tuned_optimal/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'position.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[:-1]
            for val in data:
                a.append(float(val))
            values.append(a)

    plt.figure(constrained_layout=True)
    x_axis = np.linspace(1,10000,10000)
    for ind,i in enumerate(values):
        plt.plot(x_axis, i, label=pathNames[ind], color=COLORS[ind])

    plt.xlabel("Timesteps", fontsize=16,  color='black')
    plt.ylabel("Average Position", fontsize=16,  color='black')
    plt.tick_params(axis='y', labelsize=14, labelcolor='black', width=2)
    plt.tick_params(axis='x', labelsize=14, labelcolor='black', width=2)
    plt.xlim(0,10000)
    plt.legend(prop={'size': 13}, ncols=3, bbox_to_anchor=(1,1), loc="upper right")
    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(4000))
    for label in plt.gca().xaxis.get_minorticklabels():
        label.set_fontsize(14)
    plt.tight_layout()
    # plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.savefig(os.getcwd() + "/new_data_compil/tuned_2/{}_spoofer".format(valPath) + "/position.pdf")

    # values = []
    # for path in paths:
    #     with open(os.path.join(os.getcwd(), 'tuned_optimal/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'static_position.txt'), 'r') as file:
    #         a = []
    #         data = file.read()
    #         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
    #         data = data.split()[:-1]
    #         for val in data:
    #             a.append(float(val))
    #         values.append(a)

    # plt.figure(constrained_layout=True)
    # x_axis = np.linspace(1,10000,10000)
    # for ind,i in enumerate(values):
    #     plt.plot(x_axis, i, label=pathNames[ind])

    # plt.xlabel("Timesteps")
    # plt.ylabel("Position")
    # plt.legend()
    # plt.savefig(os.getcwd() + "/new_data_compil/new_tuned_optimal/{}_spoofer".format(valPath) + "/static_position.png")


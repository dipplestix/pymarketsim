import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker
from colors import COLORS

OVERALL_PATHS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
# OVERALL_PATHS = ["A1"]
def run(valPath):
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # paths = ["A1"]
    env_above_best = []
    for path in paths:
        with open(os.path.join(os.getcwd(), 'official_rl_1optimal', '{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'env_above_best.txt'), 'r') as file:
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
        plt.plot(x_axis, y, zorder=100, label=OVERALL_PATHS[ind], color=COLORS[ind])

    # plt.xlabel("Timesteps")
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Timesteps', fontsize=18, color='black')
    plt.ylabel("Average Offset", fontsize=18, color='black')
    plt.tick_params(axis='x', labelsize=16)
    # plt.yaxis.get_offset_text().set_fontsize(16)
    # plt.rc('font', size=16)
    plt.tick_params(axis='y', labelsize=16, labelcolor='black', width=2)
    plt.xlim(1000, max(x_axis))
    ax = plt.gca()
    ax.set_xticks([1000, 4000, 7000, 10000])
    ax.set_xticklabels(['1000', '4000', '7000', '10000']) 
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)  # Optionally set y-ticks fontsize as well
    # lgd = plt.legend(ncol=3, prop={'size': 15}, bbox_to_anchor=(1,1), loc="upper right")
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/new_data_compil/rl_opt/{}_spoofer".format(valPath) + "/rounded_env_above_best.pdf")

# def run_below(valPath):
#     paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
#     # paths = ["A1"]

#     env_below_best = []
#     for path in paths:
#         with open(os.path.join(os.getcwd(), 'tuned_optimal/{}_spoofer'.format(valPath), '{}/graphs'.format(path), "env_below_best.txt"), 'r') as file:
#             a = []
#             data = file.read()
#             data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#             data = data.split()[3:]
#             data = data[:-1]
#             for val in data:
#                 if val == "nan":
#                     a.append(np.nan)
#                 else:
#                     a.append(float(val))
#             env_below_best.append(a)

#     plt.figure(constrained_layout=True)
#     x_axis = np.linspace(1,10000,10000)
#     for ind,i in enumerate(env_below_best):
#         # plt.plot(x_axis, i, label=paths[ind])
#         y = savgol_filter(i, 100, 3)
#         plt.plot(x_axis, y, zorder=100, label=OVERALL_PATHS[ind])

#     plt.xlabel("Timesteps")
#     plt.ticklabel_format(useOffset=False)
#     # plt.ticklabel_format(style='plain')
#     plt.ylabel("Order Price - Best Buy")
#     plt.legend(ncols=3)
#     plt.savefig(os.getcwd() + "/new_data_compil/new_tuned_optimal/{}_spoofer".format(valPath) + "/rounded_env_below_best.png")


valsPath = ["2e2", "6e3"]
valsPath = ["2e2"]
for valPath in valsPath:
    run(valPath)
    # run_below(valPath)
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter

OVERALL_PATHS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
# OVERALL_PATHS = ["A1"]
def run(valPath):
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # paths = ["A1"]
    midprices = []
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'xw_spoofer_2/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'midprice.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[2:]
            data = data[:-1]
            for val in data:
                a.append(float(val))
            midprices.append(a)

    x_axis_pre = np.linspace(1,1000,1000)
    x_axis_post = np.linspace(1001,10000,9000)
    # noise_constant = 100
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 3])

    # First subplot (first 2/5 of the figure)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for ind,i in enumerate(midprices):
        # val = [1e5] * noise_constant
        # val.extend(i[noise_constant:])
        # print(val)
        # plt.plot(x_axis, i, label=paths[ind])
        # plt.plot(x_axis, val, label=OVERALL_PATHS[ind])

        # First subplot (first third of the figure)
        # i[0:10] = [1e5] * 10
        ax1.plot(x_axis_pre[50:1000], i[50:1000], label=paths[ind])
        # Second subplot (second two-thirds of the figure)
        ax2.plot(x_axis_post, i[1000:], label=paths[ind])

    # ax1.set_title('Mean Midprice Pre-Spoofer Entry')
    ax1.set_xlabel('Timesteps', fontsize=20,  color='black')
    ax1.set_ylabel('Mean Midprice', fontsize=20, color='black')
    ax1.tick_params(axis='x', labelsize=18)
    ax1.yaxis.get_offset_text().set_fontsize(16)
    ax1.tick_params(axis='y', labelsize=18, labelcolor='black', width=2)
    ax1.set_xlim(0,1000)

    ax2.set_xlabel('Timesteps', fontsize=20, color='black')
    ax2.tick_params(axis='x', labelsize=18)
    ax2.yaxis.get_offset_text().set_fontsize(16)
    ax2.tick_params(axis='y', labelsize=18, labelcolor='black', width=2)
    ax2.set_xlim(1000, max(x_axis_post))
    # ax2.xaxis.set_major_locator(MultipleLocator(3000))
    # plt.xlabel("Timesteps")
    # plt.ticklabel_format(useOffset=False)
    # plt.ticklabel_format(style='plain')
    # plt.ylabel("Price")
    lgd = plt.legend(ncol=3, prop={'size': 16})
    # plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.getcwd() + "/new_data_compil/xw_2/{}_spoofer".format(valPath) + "/midprice.pdf")

    
    # midprices = []
    # for path in paths:
    #     with open(os.path.join(os.getcwd(), 'tuned_optimal/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'sim_midprice.txt'), 'r') as file:
    #         a = []
    #         data = file.read()
    #         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
    #         data = data.split()[2:]
    #         data = data[:-1]
    #         for val in data:
    #             a.append(float(val))
    #         midprices.append(a)

    # plt.figure(constrained_layout=True)
    # x_axis = np.linspace(1,10000,10000)
    # for ind,i in enumerate(midprices):
    #     # plt.plot(x_axis, i, label=paths[ind])
    #     plt.plot(x_axis, i, zorder=100, label=OVERALL_PATHS[ind])

    # plt.xlabel("Timesteps")
    # plt.ticklabel_format(useOffset=False)
    # # plt.ticklabel_format(style='plain')
    # plt.ylabel("Price")
    # plt.legend(ncol=3)
    # plt.savefig(os.getcwd() + "/new_data_compil/new_tuned_optimal/{}_spoofer".format(valPath) + "/sim_midprice.png")



valsPath = ["2e2", "6e3"]
valsPath = ["2e2"]
for valPath in valsPath:
    run(valPath)
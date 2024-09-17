import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker
from colors import COLORS
from matplotlib.ticker import ScalarFormatter

OVERALL_PATHS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
# OVERALL_PATHS = ["A1"]
def run(valPath):
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # paths = ["A1"]
    midprices = []
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'tuned_optimal/2e2_spoofer', '{}/graphs'.format(path), 'midprice.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[2:]
            data = data[:-1]
            for val in data:
                a.append(float(val))
            midprices.append(a)

    # plt.figure(figsize=(10,8))
    plt.figure()
    # plt.rcParams.update({'figure.autolayout': True})
    x_axis = np.linspace(1000,10000,9000)
    # noise_constant = 50
    for ind,i in enumerate(midprices):
        # i[:noise_constant] = [1e5] * noise_constant 
        # val = [1e5] * noise_constant
        # val.extend(i[noise_constant:])
        # print(val)
        plt.plot(x_axis, i[1000:], label=paths[ind], color=COLORS[ind])
        # plt.plot(x_axis, val, label=OVERALL_PATHS[ind])

        # First subplot (first third of the figure)
        # i[0:10] = [1e5] * 10
        # plt.plot(x_axis, i[1000:], label=paths[ind], color=COLORS[ind])

    # ax1.set_title('Mean Midprice Pre-Spoofer Entry')
    plt.xlabel('Timesteps', fontsize=20)
    plt.ylabel('Average Market Price', fontsize=20)
    # plt2.set_title('Mean Midprice Post-Spoofer Entry')
    # plt.set_xlabel('Timesteps')
    # plt.xlabel("Timesteps")
    # plt.ticklabel_format(useOffset=False)
    # plt.ticklabel_format(style='plain')
    # plt.ylabel("Price")
    ax = plt.gca()
    # ax.yaxis.get_offset_text().set_fontsize(17) 
    plt.xlim(1000,max(x_axis))
    # plt.xlim(0,max(x_axis))

    # major_ticks = ticker.MultipleLocator(3000)

    # Include a tick at 1000
    # ax.set_yticks([17, 0, -17, -40, -60])
    # ax.set_yticklabels(['17', '0', '-17', '-40', '-60']) 
    
    # ax.set_xticks([1000, 4000, 7000, 10000])
    ax.set_xticks([1000, 4000, 7000, 10000])
    # ax.set_xticklabels(['0', '2000', '4000', '6000', '8000', '10000'])
    ax.set_xticklabels(['1000', '4000', '7000', '10000']) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)  # Optionally set y-ticks fontsize as well
    # Set legend fontsize
    # plt.legend(ncol=3, fontsize=17)
    # Adjust the plot area to make room for the legend
    # plt.subplots_adjust(top=0.9)
    # plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.gca().get_yaxis().get_offset_text().set_visible(False)
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/new_data_compil/new_tuned_optimal" + "/abstract_midprice.pdf")

    # fig_legend_only_first_3 = plt.figure(figsize=(2, 2))
    # fig_legend_only_first_3 = plt.figure()
    # # Create the legend based on the first subplot
    # handles, labels = ax.get_legend_handles_labels()
    # legend_only_first_3 = fig_legend_only_first_3.legend(handles[:9], labels[:9], loc='center', ncols=3)

    # # Remove axes
    # fig_legend_only_first_3.gca().set_axis_off()

    # # Save the legend as a PNG file
    # fig_legend_only_first_3.savefig(os.getcwd() + "/new_data_compil/no_spoof_2" + "/legend.pdf", bbox_inches='tight', pad_inches=0.1)


    
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
import numpy as np
import matplotlib.pyplot as plt
import os
from colors import VALUECOMPARE, SAMPLE, SAMPLE_MIXED

# valsPath = ["2e2", "6e3"]
valsPath = ["2e2"]
# valsPath = ["6e3"]
for valPath in valsPath:
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # paths = ["A1"]

    pathNames = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # pathNames = ["A1"]

    tunedVals = []
    stdTunedVals = []
    sampleVals = []
    stdSampleVals = []

    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'official_rl_1optimal/2e2_spoofer', '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
            tempVals = []
            tempStd = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            dataStatic = data.split()[10:]
            data = data.split()[1:9]
            for ind, val in enumerate(data):
                if ind <= 3:
                    tempVals.append(float(val))
                else:
                    tempStd.append(float(val))
            tunedVals.append(tempVals)
            stdTunedVals.append(tempStd)
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'sample_full_runs/2e2_spoofer', '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
            tempVals = []
            tempStd = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            dataStatic = data.split()[10:]
            data = data.split()[1:9]
            for ind, val in enumerate(data):
                if ind <= 3:
                    tempVals.append(float(val))
                else:
                    tempStd.append(float(val))
            sampleVals.append(tempVals)
            stdSampleVals.append(tempStd)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    width = 0.35

    color_set1 = SAMPLE[0]
    color_set2 = SAMPLE[1]
    # color_set3 = SAMPLE_MIXED[0]

    for ind,i in enumerate(sampleVals):
        # Plotting the bars
        bar1 = ax.bar(ind - width/2, i[3], width=width, label="Constrained Tuned", color=color_set1)
        bar2 = ax.bar(ind + width/2, tunedVals[ind][3], width=width, label='RL', color=color_set2)

    ax.set_xlabel('Market Configuration', fontsize=20,  color='black')
    ax.set_ylabel('Spoofer Surplus', fontsize=20,  color='black')
    # ax.set_title('ZI Agent Baseline Surplus Difference')
    ax.set_xticks(np.arange(len(pathNames)))
    handles, labels = ax.get_legend_handles_labels()
    ax.tick_params(axis='y', labelsize=18, labelcolor='black', width=2)
    ax.legend(handles[:2], labels[:2], prop={'size': 16})
    ax.set_xticklabels(pathNames, fontsize=18,  color='black')
    plt.savefig(os.getcwd() + "/new_data_compil/sample/2e2_spoofer" + "/spoofer.pdf")


    fig, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    # width = 0.35
    for ind,i in enumerate(sampleVals):
        # Plotting the bars
        a = np.subtract(tunedVals[ind], i)
        bar1 = ax2.bar(ind, a[0], color=color_set1)
        # bar1 = ax2.bar(ind - width/2, i[0], width=width, label="Constrained Tuned", color=color_set1)
        # bar2 = ax2.bar(ind + width/2, tunedVals[ind][0], width=width, label='Tuned', color=color_set2)

    ax2.set_xlabel('Market Configuration', fontsize=20,  color='black')
    ax2.set_ylabel('ZI Surplus - Baseline', fontsize=20,  color='black')
    # ax.set_title('ZI Agent Baseline Surplus Difference')
    ax2.set_xticks(np.arange(len(pathNames)))
    handles, labels = ax.get_legend_handles_labels()
    ax2.tick_params(axis='y', labelsize=18, labelcolor='black', width=2)
    ax2.legend(handles[:2], labels[:2], prop={'size': 15})
    ax2.set_xticklabels(pathNames, fontsize=18,  color='black')
    plt.savefig(os.getcwd() + "/new_data_compil/sample/2e2_spoofer" + "/zi.pdf")

    fig, ax3 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    # width = 0.35
    for ind,i in enumerate(sampleVals):
        # Plotting the bars
        a = np.subtract(tunedVals[ind], i)
        bar1 = ax3.bar(ind, a[1], color=color_set1)
        # bar1 = ax3.bar(ind - width/2, i[1], width=width, label="Constrained Tuned", color=color_set1)
        # bar2 = ax3.bar(ind + width/2, tunedVals[ind][1], width=width, label='Tuned', color=color_set2)

    ax3.set_xlabel('Market Configuration', fontsize=20,  color='black')
    ax3.set_ylabel('HBL Surplus - Baseline', fontsize=20,  color='black')
    # ax.set_title('ZI Agent Baseline Surplus Difference')
    ax3.set_xticks(np.arange(len(pathNames)))
    handles, labels = ax.get_legend_handles_labels()
    ax3.tick_params(axis='y', labelsize=18, labelcolor='black', width=2)
    ax3.legend(handles[:2], labels[:2], prop={'size': 15})
    ax3.set_xticklabels(pathNames, fontsize=18,  color='black')
    plt.savefig(os.getcwd() + "/new_data_compil/sample/2e2_spoofer" + "/hbl.pdf")



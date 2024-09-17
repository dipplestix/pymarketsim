import numpy as np
import matplotlib.pyplot as plt
import os
from colors import VALUECOMPARE, SAMPLE_MIXED

# valsPath = ["2e2", "6e3"]
valsPath = ["2e2"]
# valsPath = ["2e2", "6e3"]
for valPath in valsPath:
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # paths = ["A1"]

    pathNames = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # pathNames = ["A1"]

    baselineVals = []
    stdBaselineVals = []
    tunedVals = []
    stdTunedVals = []
    rlVals = []
    stdRlVals = []
    staticVals = []
    stdStaticVals = []
    
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'official_rl_1optimal/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
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
            rlVals.append(tempVals)
            stdRlVals.append(tempStd)
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'sample_full_runs/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
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

    color_set2 = SAMPLE_MIXED[0]
    color_set3 = VALUECOMPARE[1]
    width=0.35

    fig, ax3 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for ind, i in enumerate(rlVals):
        bar2Spoof = ax3.bar(ind, tunedVals[ind][3], width, label='Constrained Tuned', color=color_set2)
        bar3Spoof = ax3.bar(ind + width, i[3], width, label='Learning', color=color_set3)

    ax3.set_xlabel('Market Configuration', fontsize=22,  color='black')
    ax3.set_ylabel('Spoofer Surplus', fontsize=22,  color='black')
    ax3.tick_params(axis='y', labelsize=20, labelcolor='black', width=2)
    # ax2.set_title('HBL Agent Baseline Surplus Difference')
    ax3.set_xticks(np.arange(len(pathNames)))
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles[:2], labels[:2], prop={'size': 17})
    ax3.set_xticklabels(pathNames, fontsize=20,  color='black')

    plt.savefig(os.getcwd() + "/new_data_compil/s_eff/{}_spoofer".format(valPath) + "/spoofer.pdf")



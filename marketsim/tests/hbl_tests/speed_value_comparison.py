import numpy as np
import matplotlib.pyplot as plt
import os
from colors import VALUECOMPARE
# valsPath = ["2e2", "6e3"]
valsPath = ["2e2"]
# valsPath = ["6e3"]
for valPath in valsPath:
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # paths = ["A1"]

    pathNames = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    # pathNames = ["A1"]

    val6e3Tuned = []
    std6e3Tuned = []
    val6e3RlVals = []
    std6e3RlVals = []
    val6e3StaticVals = []
    std6e3StaticVals = []
    tunedVals = []
    stdTunedVals = []
    rlVals = []
    stdRlVals = []
    staticVals = []
    stdStaticVals = []
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'official_rl_1optimal/6e3_spoofer', '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
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
            val6e3RlVals.append(tempVals)
            std6e3RlVals.append(tempStd)

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
            rlVals.append(tempVals)
            stdRlVals.append(tempStd)
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'tuned_optimal/2e2_spoofer', '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
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
        with open(os.path.join(os.getcwd(), 'tuned_optimal/6e3_spoofer', '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
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
            val6e3Tuned.append(tempVals)
            std6e3Tuned.append(tempStd)
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'xw_spoofer_2/2e2_spoofer', '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
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
            staticVals.append(tempVals)
            stdStaticVals.append(tempStd)
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'xw_spoofer_2/6e3_spoofer', '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
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
            val6e3StaticVals.append(tempVals)
            std6e3StaticVals.append(tempStd)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    width = 0.25

    color_set1 = VALUECOMPARE[0]
    color_set2 = VALUECOMPARE[1]
    color_set3 = VALUECOMPARE[2]

    for ind,i in enumerate(tunedVals):
        # Plotting the bars
        static = np.divide(val6e3StaticVals[ind], staticVals[ind])
        rl = np.divide(val6e3RlVals[ind], rlVals[ind])
        tuned = np.divide(val6e3Tuned[ind], tunedVals[ind])
        
        bar1 = ax.bar(ind - width, static[0], width, label="Static", color=color_set1)
        bar2 = ax.bar(ind, rl[0], width, label='RL', color=color_set2)
        bar3 = ax.bar(ind + width, tuned[0], width, label='Tuned', color=color_set3)

    ax.set_xlabel('Market Configuration', fontsize=18,  color='black')
    ax.set_ylabel('Proportion of Surplus Achieved', fontsize=18,  color='black')
    # ax.set_title('ZI Agent Baseline Surplus Difference')
    ax.set_xticks(np.arange(len(pathNames)))
    handles, labels = ax.get_legend_handles_labels()
    ax.tick_params(axis='y', labelsize=18, labelcolor='black', width=2)
    ax.legend(handles[:3], labels[:3], prop={'size': 15})
    ax.set_xticklabels(pathNames, fontsize=18,  color='black')
    plt.savefig(os.getcwd() + "/new_data_compil/combined/combined_comp" + "/zi.pdf")


    fig, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for ind, i in enumerate(tunedVals):
        static = np.divide(val6e3StaticVals[ind], staticVals[ind])
        rl = np.divide(val6e3RlVals[ind], rlVals[ind])
        tuned = np.divide(val6e3Tuned[ind], tunedVals[ind])

        barHBL = ax2.bar(ind - width, static[1], width, label='Static', color=color_set1)
        bar2HBL = ax2.bar(ind, rl[1], width, label='RL', color=color_set2)
        bar3HBL = ax2.bar(ind + width, tuned[1], width, label='Tuned', color=color_set3)

    ax2.set_xlabel('Market Configuration', fontsize=18,  color='black')
    ax2.set_ylabel('Proportion of Surplus Achieved', fontsize=18,  color='black')
    ax2.tick_params(axis='y', labelsize=18, labelcolor='black', width=2)
    # ax2.set_title('HBL Agent Baseline Surplus Difference')
    ax2.set_xticks(np.arange(len(pathNames)))
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:3], labels[:3], prop={'size': 15})
    ax2.set_xticklabels(pathNames, fontsize=18,  color='black')

    plt.savefig(os.getcwd() + "/new_data_compil/combined/combined_comp" + "/hbl.pdf")

    fig, ax3 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for ind, i in enumerate(tunedVals):
        static =  np.divide(val6e3StaticVals[ind], staticVals[ind])
        rl = np.divide(val6e3RlVals[ind], rlVals[ind])
        tuned = np.divide(val6e3Tuned[ind], tunedVals[ind])

        barSpoof = ax3.bar(ind - width, static[3], width, label='Static', color=color_set1)
        bar2Spoof = ax3.bar(ind, rl[3], width, label='RL', color=color_set2) 
        bar3Spoof = ax3.bar(ind + width, tuned[3], width, label='Tuned', color=color_set3)


    ax3.set_xlabel('Market Configuration', fontsize=18,  color='black')
    ax3.set_ylabel('Proportion of Surplus Achieved', fontsize=18,  color='black')
    ax3.tick_params(axis='y', labelsize=18, labelcolor='black', width=2)
    
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles[:3], labels[:3], prop={'size': 15})
    ax3.set_xticks(np.arange(len(pathNames)))
    ax3.set_xticklabels(pathNames, fontsize=18,  color='black')

    plt.savefig(os.getcwd() + "/new_data_compil/combined/combined_comp" + "/spoofer.pdf")

    fig, ax4 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for ind, i in enumerate(tunedVals):
        static = np.divide(val6e3StaticVals[ind], staticVals[ind])
        rl = np.divide(val6e3RlVals[ind], rlVals[ind])
        tuned = np.divide(val6e3Tuned[ind], tunedVals[ind])

        barHBL = ax4.bar(ind - width, static[2], width, label='Static', color=color_set1)
        bar2HBL = ax4.bar(ind, rl[2], width, label='RL', color=color_set2)
        bar3HBL = ax4.bar(ind + width, tuned[2], width, label='Tuned', color=color_set3)

    ax4.set_xlabel('Market Configuration', fontsize=18,  color='black')
    ax4.set_ylabel('Proportion of Surplus Achieved', fontsize=18,  color='black')
    ax4.tick_params(axis='y', labelsize=18, labelcolor='black', width=2)
    # ax2.set_title('HBL Agent Baseline Surplus Difference')
    ax4.set_xticks(np.arange(len(pathNames)))
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles[:3], labels[:3], prop={'size': 15})
    ax4.set_xticklabels(pathNames, fontsize=18,  color='black')

    plt.savefig(os.getcwd() + "/new_data_compil/combined/combined_comp" + "/mm.pdf")


    a = np.subtract(np.array(tunedVals)[:, 3], np.array(val6e3Tuned)[:, 3])
    print(a)
    # for arr in a:
    #     print(np.mean(arr))


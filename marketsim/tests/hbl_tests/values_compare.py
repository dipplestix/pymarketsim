import numpy as np
import matplotlib.pyplot as plt
import os
from colors import VALUECOMPARE

# valsPath = ["2e2", "6e3"]
valsPath = ["2e2"]
valsPath = ["6e3"]
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
        with open(os.path.join(os.getcwd(), 'no_spoof', '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
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
            baselineVals.append(tempVals)
        
        with open(os.path.join(os.getcwd(), 'no_spoof', '{}/graphs'.format(path), 'avgstd.txt'), 'r') as file:
            tempVals = []
            tempStd = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[4:]
            for ind, val in enumerate(data):
                tempStd.append(float(val))
            stdBaselineVals.append(tempStd)
    
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
            # stdRlVals.append(tempStd)
    
        with open(os.path.join(os.getcwd(), 'official_rl_1optimal/2e2_spoofer', '{}/graphs'.format(path), 'avgstd.txt'), 'r') as file:
            tempVals = []
            tempStd = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[4:]
            for ind, val in enumerate(data):
                tempStd.append(float(val))
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
            # stdTunedVals.append(tempStd)
    
        with open(os.path.join(os.getcwd(), 'tuned_optimal/2e2_spoofer', '{}/graphs'.format(path), 'avgstd.txt'), 'r') as file:
            tempVals = []
            tempStd = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[4:]
            for ind, val in enumerate(data):
                tempStd.append(float(val))
            stdTunedVals.append(tempStd)
    
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
            # stdStaticVals.append(tempStd)
    
        with open(os.path.join(os.getcwd(), 'xw_spoofer_2/2e2_spoofer', '{}/graphs'.format(path), 'avgstd.txt'), 'r') as file:
            tempVals = []
            tempStd = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[4:]
            for ind, val in enumerate(data):
                tempStd.append(float(val))
            stdStaticVals.append(tempStd)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    width = 0.25

    print("VALs", baselineVals)
    print("STD", stdBaselineVals)
    print("VALS2", rlVals)
    print("STD2", stdRlVals)
    print("STD3", stdStaticVals)

    color_set1 = VALUECOMPARE[0]
    color_set2 = VALUECOMPARE[1]
    color_set3 = VALUECOMPARE[2]

    for ind,i in enumerate(baselineVals):
        # Plotting the bars
        static = np.subtract(staticVals[ind], baselineVals[ind])
        rl = np.subtract(rlVals[ind], baselineVals[ind])
        tuned = np.subtract(tunedVals[ind], baselineVals[ind])
        
        bar1 = ax.bar(ind - width, static[0], width=width, yerr=stdStaticVals[ind][0]/np.sqrt(40000), capsize=2, error_kw=dict(lw=1, capthick=1), label="Baseline", color=color_set1)
        bar2 = ax.bar(ind, rl[0], width=width,             yerr=stdRlVals[ind][0]/np.sqrt(40000), capsize=2, error_kw=dict(lw=1, capthick=1), label='R-Learned' , color=color_set2)
        bar3 = ax.bar(ind + width, tuned[0], width=width,  yerr=stdTunedVals[ind][0]/np.sqrt(40000), capsize=2, error_kw=dict(lw=1, capthick=1), label='Tuned', color=color_set3)

    ax.set_xlabel('Market Configuration', fontsize=22,  color='black')
    ax.set_ylabel('Average Difference in ZI Surplus', fontsize=22,  color='black')
    # ax.set_title('ZI Agent Baseline Surplus Difference')
    ax.set_xticks(np.arange(len(pathNames)))
    # handles, labels = ax.get_legend_handles_labels()
    ax.tick_params(axis='y', labelsize=20, labelcolor='black', width=2)
    # ax.legend(handles[:3], labels[:3], prop={'size': 17})
    ax.set_xticklabels(pathNames, fontsize=20,  color='black')
    # plt.savefig(os.getcwd() + "/new_data_compil/combined/2e2_spoofer" + "/zi.pdf")


    fig, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for ind, i in enumerate(baselineVals):
        static = np.subtract(staticVals[ind], baselineVals[ind])
        rl = np.subtract(rlVals[ind], baselineVals[ind])
        tuned = np.subtract(tunedVals[ind], baselineVals[ind])

        barHBL = ax2.bar(ind - width, static[1],   yerr=stdStaticVals[ind][1]/np.sqrt(40000), capsize=2, error_kw=dict(lw=1, capthick=1), width=width, label='Static', color=color_set1)
        bar2HBL = ax2.bar(ind, rl[1], width=width, yerr=stdRlVals[ind][1]/np.sqrt(40000), capsize=2, error_kw=dict(lw=1, capthick=1), label='RL', color=color_set2)
        bar3HBL = ax2.bar(ind + width, tuned[1],   yerr=stdTunedVals[ind][1]/np.sqrt(40000), capsize=2, error_kw=dict(lw=1, capthick=1), width=width, label='Tuned', color=color_set3)

    ax2.set_xlabel('Market Configuration', fontsize=25,  color='black')
    ax2.set_ylabel('Average Surplus Difference', fontsize=25,  color='black')
    ax2.tick_params(axis='y', labelsize=23, labelcolor='black', width=2)
    # ax2.set_title('HBL Agent Baseline Surplus Difference')
    ax2.set_xticks(np.arange(len(pathNames)))
    # handles, labels = ax2.get_legend_handles_labels()
    # ax2.legend(handles[:3], labels[:3], prop={'size': 17})
    ax2.set_xticklabels(pathNames, fontsize=23,  color='black')

    plt.savefig(os.getcwd() + "/new_data_compil/combined/2e2_spoofer" + "/abstract_hbl.pdf")

    fig, ax3 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for ind, i in enumerate(baselineVals):
        static = np.subtract(staticVals[ind], baselineVals[ind])
        rl = np.subtract(rlVals[ind], baselineVals[ind])
        tuned = np.subtract(tunedVals[ind], baselineVals[ind])
        barSpoof = ax3.bar(ind - width, static[3],   yerr=stdStaticVals[ind][3]/np.sqrt(40000), capsize=2, error_kw=dict(lw=1, capthick=1), width=width, label='Static', color=color_set1)
        bar2Spoof = ax3.bar(ind, rl[3], width=width, yerr=stdRlVals[ind][3]/np.sqrt(40000), capsize=2, error_kw=dict(lw=1, capthick=1), label='RL', color=color_set2)
        bar3Spoof = ax3.bar(ind + width, tuned[3],   yerr=stdTunedVals[ind][3]/np.sqrt(40000), capsize=2, error_kw=dict(lw=1, capthick=1), width=width, label='Tuned', color=color_set3)


    ax3.set_xlabel('Market Configuration', fontsize=25,  color='black')
    ax3.set_ylabel('Average Surplus', fontsize=25,  color='black')
    ax3.tick_params(axis='y', labelsize=23, labelcolor='black', width=2)
    
    # handles, labels = ax3.get_legend_handles_labels()
    # ax3.legend(handles[:3], labels[:3], prop={'size': 17})
    ax3.set_xticks(np.arange(len(pathNames)))
    ax3.set_xticklabels(pathNames, fontsize=23,  color='black')

    plt.savefig(os.getcwd() + "/new_data_compil/combined/2e2_spoofer" + "/abstract_spoofer.pdf")

    fig, ax4 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for ind, i in enumerate(baselineVals):
        static = np.subtract(staticVals[ind], baselineVals[ind])
        rl = np.subtract(rlVals[ind], baselineVals[ind])
        tuned = np.subtract(tunedVals[ind], baselineVals[ind])

        barHBL = ax4.bar(ind - width, static[2], width=width, label='Static', color=color_set1)
        bar2HBL = ax4.bar(ind, rl[2], width=width, label='RL', color=color_set2)
        bar3HBL = ax4.bar(ind + width, tuned[2], width=width, label='Tuned', color=color_set3)

    ax4.set_xlabel('Market Configuration', fontsize=22,  color='black')
    ax4.set_ylabel('MM Surplus - Baseline', fontsize=22,  color='black')
    ax4.tick_params(axis='y', labelsize=20, labelcolor='black', width=2)
    # ax2.set_title('HBL Agent Baseline Surplus Difference')
    ax4.set_xticks(np.arange(len(pathNames)))
    # handles, labels = ax4.get_legend_handles_labels()
    # ax4.legend(handles[:3], labels[:3], prop={'size': 17})
    ax4.set_xticklabels(pathNames, fontsize=20,  color='black')

    # plt.savefig(os.getcwd() + "/new_data_compil/combined/2e2_spoofer" + "/mm.pdf")


    fig_legend_only_first_3 = plt.figure(figsize=(2, 2))
    # Create the legend based on the first subplot
    handles, labels = ax.get_legend_handles_labels()
    legend_only_first_3 = fig_legend_only_first_3.legend(handles[:3], labels[:3], loc='center', prop={'size': 17}, ncols=3)

    # Remove axes
    fig_legend_only_first_3.gca().set_axis_off()

    # Save the legend as a PNG file
    # fig_legend_only_first_3.savefig(os.getcwd() + "/new_data_compil/combined/2e2_spoofer" + "/legend.pdf", bbox_inches='tight', pad_inches=0.1)



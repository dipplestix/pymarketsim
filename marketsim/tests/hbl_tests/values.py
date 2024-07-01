import numpy as np
import matplotlib.pyplot as plt
import os

valsPath = ["4e2", "2e2"]
for valPath in vals:
    paths = ["A1", "A2", "A3", "C1", "C2", "C3"]
    if valPath == "4e2":
        pathsSupp = ["A1_rerun", "A2", "A3"]
    else:
        pathsSupp = ["A1", "A2", "A3"]

    pathNames = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

    values = []
    valuesStatic = []
    for path in pathsSupp:
        with open(os.path.join(os.getcwd(), 'supplemental_exps/rl/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'values.txt'), 'r') as file:
            a = []
            staticA = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            dataStatic = data.split()[34:]
            data = data.split()[5:31]
            print(data)
            print(dataStatic)
            for val in data:
                a.append(float(val))
            for val in dataStatic:
                staticA.append(float(val))
            values.append(a)
            valuesStatic.append(staticA)

    for path in paths:
        with open(os.path.join(os.getcwd(), 'official_rl_exps/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'values.txt'), 'r') as file:
            a = []
            staticA = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            dataStatic = data.split()[34:]
            data = data.split()[5:31]
            print(data)
            print(dataStatic)
            for val in data:
                a.append(float(val))
            for val in dataStatic:
                staticA.append(float(val))
            values.append(a)
            valuesStatic.append(staticA)

    markers = {'Spoofer': 'o', 'ZI': '^', 'HBL': 'x'}
    colors = {"Spoofer": 'blue', 'ZI':"green", 'HBL': 'orange'}

    plt.figure(constrained_layout=True)
    for ind,i in enumerate(values):
        plt.scatter(ind, np.mean(i[:12]), marker=markers["ZI"], color=colors["ZI"], label="4e-2 ZI")
        # plt.errorbar(ind, np.mean(i[:12]), yerr=np.std(i[:12]), capsize=5, fmt=markers["ZI"], color=colors["ZI"], label="ZI")
        plt.scatter(ind, np.mean(i[12:24]), marker=markers["HBL"], color=colors["HBL"], label="4e-2 HBL")
        # plt.errorbar(ind, np.mean(i[12:24]), yerr=np.std(i[:12]), capsize=5, fmt=markers["HBL"], color=colors["HBL"], label="HBL")
        plt.scatter(ind, np.mean(i[-1]), marker=markers["Spoofer"], color=colors["Spoofer"], s=10, label="4e-2 Spoofer")
        # plt.errorbar(ind, np.mean(i[12:24]), yerr=np.stdev(i[:12]), capsize=5)

    plt.xticks(ticks=np.linspace(0,len(values), 9,endpoint=False), labels=pathNames)
    plt.xlabel("Market Configuration")
    plt.ylabel("Surplus")
    handles, labels = plt.gca().get_legend_handles_labels()
    # input(labels)
    # newHandles = [handles[:3],handles[-3:]]
    # newLabels = [labels[:3], labels[:-3:]]
    # input(handles[-3:])
    newHandles = handles[:3]
    # newHandles.extend(handles[-3:])
    newLabels = labels[:3]
    # newLabels.extend(labels[-3:])
    plt.legend(newHandles, newLabels)
    # print(os.getcwd())
    plt.savefig(os.getcwd() + "/data_compil/{}".format(valPath) + "/surplus_chart.png")


    plt.figure(constrained_layout=True)
    for ind,i in enumerate(valuesStatic):
        plt.scatter(ind, np.mean(i[:12]), marker=markers["ZI"], color=colors["ZI"], label="Static ZI")
        # plt.errorbar(ind, np.mean(i[:12]), yerr=np.std(i[:12]), capsize=5, fmt=markers["ZI"], color=colors["ZI"], label="ZI")
        plt.scatter(ind, np.mean(i[12:24]), marker=markers["HBL"], color=colors["HBL"], label="Static HBL")
        # plt.errorbar(ind, np.mean(i[12:24]), yerr=np.std(i[:12]), capsize=5, fmt=markers["HBL"], color=colors["HBL"], label="HBL")
        plt.scatter(ind, np.mean(i[-1]), marker=markers["Spoofer"], color=colors["Spoofer"], s=10, label="Static Spoofer")
        # plt.errorbar(ind, np.mean(i[12:24]), yerr=np.stdev(i[:12]), capsize=5)

    plt.xticks(ticks=np.linspace(0,len(values), 9, endpoint=False), labels=pathNames)
    plt.xlabel("Market Configuration")
    plt.ylabel("Surplus")
    handles, labels = plt.gca().get_legend_handles_labels()
    # input(labels)
    # newHandles = [handles[:3],handles[-3:]]
    # newLabels = [labels[:3], labels[:-3:]]
    # input(handles[-3:])
    newHandles = handles[:3]
    # newHandles.extend(handles[-3:])
    newLabels = labels[:3]
    # newLabels.extend(labels[-3:])
    plt.legend(newHandles, newLabels)
    # print(os.getcwd())
    plt.savefig(os.getcwd() + "/data_compil/{}".format(valPath) + "/static_surplus_chart.png")





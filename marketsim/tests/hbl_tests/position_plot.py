import numpy as np
import matplotlib.pyplot as plt
import os

valsPath = ["2e2", "6e3"]
valsPath = ["2e2"]
# valsPath = ["2e2", "6e3"]
for valPath in valsPath:
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    paths = ["A1"]
    pathNames = ["A1", "A2","A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    pathNames = ["A1"]
    values = []
   
    for path in paths:
        with open(os.path.join(os.getcwd(), 'baseline_no_MM/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'position.txt'), 'r') as file:
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
        plt.plot(x_axis, i, label=pathNames[ind])

    plt.xlabel("Timesteps")
    plt.ylabel("Position")
    plt.legend()
    plt.savefig(os.getcwd() + "/new_data_compil/bl_no_mm/{}_spoofer".format(valPath) + "/position.png")

    values = []
    for path in paths:
        with open(os.path.join(os.getcwd(), 'baseline_no_MM/{}_spoofer'.format(valPath), '{}/graphs'.format(path), 'static_position.txt'), 'r') as file:
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
        plt.plot(x_axis, i, label=pathNames[ind])

    plt.xlabel("Timesteps")
    plt.ylabel("Position")
    plt.legend()
    plt.savefig(os.getcwd() + "/new_data_compil/bl_no_mm/{}_spoofer".format(valPath) + "/static_position.png")


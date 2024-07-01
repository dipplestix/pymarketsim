import numpy as np
import matplotlib.pyplot as plt
import os

valsPath = ["2e2", "6e3"]
pathToTake = [{"A1":"s", "A2":"s", "A3":"s", "B1":"s", "B2":"s", "B3":"s", "C1":"s", "C2":"s", "C3":"s"},
{"A1":"n", "A2":"s", "A3":"s", "B1":"s", "B2":"s", "B3":"n", "C1":"s", "C2":"s", "C3":"n"}
]
for indPath, valPath in enumerate(valsPath):
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

    pathNames = ["A1", "A2","A3", "B1", "B2", "B3", "C1", "C2", "C3"]

    values = []
    values_static = []
    for path in paths:
        with open(os.path.join(os.getcwd(), 'param_search/{}_optimal'.format(valPath), '{}/graphs'.format(path), 'position.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[:-1]
            for val in data:
                a.append(float(val))
            values.append(a)

        with open(os.path.join(os.getcwd(), 'param_search/{}_optimal'.format(valPath), '{}/graphs'.format(path), 'static_position.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[:-1]
            for val in data:
                a.append(float(val))
            values_static.append(a)

    plt.figure(constrained_layout=True)
    x_axis = np.linspace(1,10000,10000)
    for ind,i in enumerate(values):
        if pathToTake[indPath][pathNames[ind]] == "s":
            plot = values_static[ind]
        else:
            plot = i
        plt.plot(x_axis, plot, label=pathNames[ind])

    plt.xlabel("Timesteps")
    plt.ylabel("Position")
    plt.legend()
    plt.savefig(os.getcwd() + "/new_data_compil/param_search/{}_spoofer".format(valPath) + "/position.png")


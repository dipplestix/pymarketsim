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

 
    tunedPosition2e2 = []
    tunedPosition6e3 = []
        
    midprice2e2 = []
    midprice6e3 = []
    for path in paths:
        with open(os.path.join(os.getcwd(), 'tuned_optimal/2e2_spoofer', '{}/graphs'.format(path), 'position.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[:-1]
            for val in data:
                a.append(float(val))
            tunedPosition2e2.append(a[-1])
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'tuned_optimal/6e3_spoofer', '{}/graphs'.format(path), 'position.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[:-1]
            for val in data:
                a.append(float(val))
            tunedPosition6e3.append(a[-1])
    print(np.subtract(tunedPosition2e2, tunedPosition6e3))

    for path in paths:
        with open(os.path.join(os.getcwd(), 'tuned_optimal/2e2_spoofer', '{}/graphs'.format(path), 'midprice.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[2:]
            data = data[:-1]
            for val in data:
                a.append(float(val))
            midprice2e2.append(max(a[1000:]) - min(a[1000:]))
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'tuned_optimal/6e3_spoofer', '{}/graphs'.format(path), 'midprice.txt'), 'r') as file:
            a = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[2:]
            data = data[:-1]
            for val in data:
                a.append(float(val))
            midprice6e3.append(max(a[1000:]) - min(a[1000:]))
    print(np.subtract(midprice2e2, midprice6e3))


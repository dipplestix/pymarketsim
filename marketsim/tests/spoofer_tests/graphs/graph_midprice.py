import numpy as np
import matplotlib.pyplot as plt
import os
from colors import COLORS

OVERALL_PATHS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
def run(valPath):
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    midprices = []
    
    for path in paths:
        #TODO: Change path accordingly
        with open(os.path.join(os.getcwd(), '[DATA PATH]', '{}/graphs'.format(path), 'midprice.txt'), 'r') as file:
            temp = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            data = data.split()[2:]
            data = data[:-1]
            for val in data:
                temp.append(float(val))
            midprices.append(temp)

    plt.figure()
    x_axis = np.linspace(1000,10000,9000)
    for ind,i in enumerate(midprices):
        plt.plot(x_axis, i[1000:], label=paths[ind], color=COLORS[ind])

    plt.xlabel('Timesteps', fontsize=20)
    plt.ylabel('Average Market Price', fontsize=20)

    ax = plt.gca()
    plt.xlim(1000,max(x_axis))
    ax.set_xticks([1000, 4000, 7000, 10000])
    ax.set_xticklabels(['1000', '4000', '7000', '10000']) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().get_yaxis().get_offset_text().set_visible(False)
    plt.tight_layout()
    #TODO
    plt.savefig(os.getcwd() + "/[DATA PATH]/no_spoof_3" + "/midprice.pdf")

run()
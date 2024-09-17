import numpy as np
import matplotlib.pyplot as plt
import os
from colors import VALUECOMPARE, VALS

# valsPath = ["2e2", "6e3"]
# valsPath = ["2e2"]
# valsPath = ["6e3"]
paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
# paths = ["A1"]

pathNames = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
# pathNames = ["A1"]

# buyStaticbuyDiff = []
# lossStaticDiff = []
# buyRLDiff = []
# lossRLDiff = []
# buyTunedDiff = []
# lossTunedDiff = []

buy2e2 = []
loss2e2 = []
buy6e3 = []
loss6e3 = []


tunedbuy2e2 = []
tunedloss2e2 = []
tunedbuy6e3 = []
tunedloss6e3 = []

# for path in paths:
#     with open(os.path.join(os.getcwd(), 'xw_spoofer_2/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#         tempBuy = []
#         tempLoss = []
#         data = file.read()
#         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#         buys = data.split()[:5]
#         losses = data.split()[5:10]
#         for buy in buys:
#             tempBuy.append(float(buy))
#         for loss in losses:
#             tempLoss.append(float(loss))
#     buy6e3.append(tempBuy)
#     loss6e3.append(tempLoss)

# for path in paths:
#     with open(os.path.join(os.getcwd(), 'xw_spoofer_2/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#         tempBuy = []
#         tempLoss = []
#         data = file.read()
#         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#         buys = data.split()[:5]
#         losses = data.split()[5:10]
#         for buy in buys:
#             tempBuy.append(float(buy))
#         for loss in losses:
#             tempLoss.append(float(loss))
#     buy2e2.append(tempBuy)
#     loss2e2.append(tempLoss)

# buyStaticDiff = np.subtract(buy6e3, buy2e2)
# lossStaticDiff = np.subtract(loss6e3, loss2e2)

# for path in paths:
#     with open(os.path.join(os.getcwd(), 'official_rl_1optimal/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#         tempBuy = []
#         tempLoss = []
#         data = file.read()
#         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#         buys = data.split()[:5]
#         losses = data.split()[5:10]
#         for buy in buys:
#             tempBuy.append(float(buy))
#         for loss in losses:
#             tempLoss.append(float(loss))
#     RLbuy2e2.append(tempBuy)
#     RLloss2e2.append(tempLoss)
        
# for path in paths:
#     with open(os.path.join(os.getcwd(), 'official_rl_1optimal/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#         tempBuy = []
#         tempLoss = []
#         data = file.read()
#         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#         buys = data.split()[:5]
#         losses = data.split()[5:10]
#         for buy in buys:
#             tempBuy.append(float(buy))
#         for loss in losses:
#             tempLoss.append(float(loss))
#     RLbuy6e3.append(tempBuy)
#     RLloss6e3.append(tempLoss)

# buyRLDiff = np.subtract(RLbuy6e3, RLbuy2e2)
# lossRLDiff = np.subtract(RLloss6e3, RLloss2e2)
# print(RLbuy6e3, "\n", RLbuy2e2)

positionFast = []
positionSlow = []

# load_from = ["o", "o", "o", "o", "o", "o", "o", "o"]

# for path in paths:
#     with open(os.path.join(os.getcwd(), 'tuned_optimal/2e2_spoofer', '{}/graphs'.format(path), 'position.txt'), 'r') as file:
#         tempBuy = []
#         tempLoss = []
#         data = file.read()
#         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#         finalPosition = data.split()[-2]
#         # losses = data.split()[5:10]
#     positionFast.append(float(finalPosition))
        
# for path in paths:
#     with open(os.path.join(os.getcwd(), 'tuned_optimal_buys/6e3_spoofer', '{}/graphs'.format(path), 'position.txt'), 'r') as file:
#         tempBuy = []
#         tempLoss = []
#         data = file.read()
#         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#         finalPosition = data.split()[-2]
#         # losses = data.split()[5:10]
#     positionSlow.append(float(finalPosition))

valueFast = []
valueSlow = []

positionFast = []
positionSlow = []

# select2e2 = ["o", "ob", "l", "ob", "f", "o", "o", "f", "ob"]
select2e2 = ["f", "ob", "o", "l", "f", "o", "ob", "f", "ob"]
select6e3 = ["f", "ob", "ob", "f", "f", "ob", "ob", "ob", "ob"]
# select2e2 = ["ob", "ob", "ob", "ob", "ob", "ob", "ob", "ob", "ob"] 
# select6e3 = ["ob", "ob", "ob", "ob", "ob", "ob", "ob", "ob", "ob"] 
# select6e3 = ["ob", "f", "f", "ob", "f", "ob", "ob", "ob", "ob", "ob"]

# for path in paths:
#     with open(os.path.join(os.getcwd(), 'tuned_optimal_buys/2e2_spoofer', '{}/graphs'.format(path), 'position.txt'), 'r') as file:
#         tempBuy = []
#         tempLoss = []
#         data = file.read()
#         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#         vals = data.split()
#         print("VLAS", vals[-1], vals[-2])
#     positionFast.append(float(vals[-2]))
        
# for path in paths:
#     with open(os.path.join(os.getcwd(), 'tuned_optimal_buys/6e3_spoofer', '{}/graphs'.format(path), 'position.txt'), 'r') as file:
#         tempBuy = []
#         tempLoss = []
#         data = file.read()
#         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#         vals = data.split()
#         print("VLAS", vals[-1], vals[-2])
#     positionSlow.append(float(vals[-2]))

for ind, path in enumerate(paths):
    with open(os.path.join(os.getcwd(), 'tuned_buys_final/2e2_spoofer', '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
        tempBuy = []
        tempLoss = []
        data = file.read()
        data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
        vals = data.split()
        # print("VLAS", vals)
    valueFast.append(float(vals[4]))
        
for path in paths:
    with open(os.path.join(os.getcwd(), 'tuned_buys_final/6e3_spoofer', '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
        tempBuy = []
        tempLoss = []
        data = file.read()
        data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
        vals = data.split()
        print(vals)
    valueSlow.append(float(vals[4]))

tunedbuy2e2std = []
tunedloss2e2std = []
tunedbuy6e3std = []
tunedloss6e3std = []

# for ind, path in enumerate(paths):
#     # if path == "A2":
#     #     a = []
#     #     with open(os.path.join(os.getcwd(), 'tuned_optimal_buys/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#     #         tempBuy = []
#     #         tempLoss = []
#     #         data = file.read()
#     #         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#     #         buys = data.split()[:5]
#     #         losses = data.split()[5:10]
#     #         for buy in buys:
#     #             tempBuy.append(float(buy))
#     #         for loss in losses:
#     #             a.append(float(loss))
#     #     with open(os.path.join(os.getcwd(), 'tuned_buys_loss/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#     #         tempBuy = []
#     #         tempLoss = []
#     #         data = file.read()
#     #         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#     #         buys = data.split()[:5]
#     #         losses = data.split()[5:10]
#     #         for buy in buys:
#     #             tempBuy.append(float(buy))
#     #         for loss in losses:
#     #             a.append(float(loss))
#     #     # with open(os.path.join(os.getcwd(), 'tuned_optimal/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#     #     #     tempBuy = []
#     #     #     tempLoss = []
#     #     #     data = file.read()
#     #     #     data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#     #     #     buys = data.split()[:5]
#     #     #     losses = data.split()[5:10]
#     #     #     for buy in buys:
#     #     #         tempBuy.append(float(buy))
#     #     #     for loss in losses:
#     #     #         a.append(float(loss))
#     #     with open(os.path.join(os.getcwd(), 'tuned_buys_final/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#     #         tempBuy = []
#     #         tempLoss = []
#     #         data = file.read()
#     #         data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#     #         buys = data.split()[:5]
#     #         losses = data.split()[5:10]
#     #         for buy in buys:
#     #             tempBuy.append(float(buy))
#     #         for loss in losses:
#     #             a.append(float(loss))
#     #     print("AASDNJKAS", a)

#     if select2e2[ind] == "ob":
#         print("HELLO")
#         with open(os.path.join(os.getcwd(), 'tuned_optimal_buys/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#             tempBuy = []
#             tempLoss = []
#             data = file.read()
#             data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#             buys = data.split()[:5]
#             losses = data.split()[5:10]
#             for buy in buys:
#                 tempBuy.append(float(buy))
#             for loss in losses:
#                 tempLoss.append(float(loss))
#     elif select2e2[ind] == "l":
#         print("LOSS")
#         with open(os.path.join(os.getcwd(), 'tuned_buys_loss/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#             tempBuy = []
#             tempLoss = []
#             data = file.read()
#             data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#             buys = data.split()[:5]
#             losses = data.split()[5:10]
#             for buy in buys:
#                 tempBuy.append(float(buy))
#             for loss in losses:
#                 tempLoss.append(float(loss))
#     elif select2e2[ind] == "o":
#         print("HERE")
#         with open(os.path.join(os.getcwd(), 'tuned_optimal/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#             tempBuy = []
#             tempLoss = []
#             data = file.read()
#             data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#             buys = data.split()[:5]
#             losses = data.split()[5:10]
#             for buy in buys:
#                 tempBuy.append(float(buy))
#             for loss in losses:
#                 tempLoss.append(float(loss))
#     else:
#         with open(os.path.join(os.getcwd(), 'tuned_buys_final/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#             tempBuy = []
#             tempLoss = []
#             data = file.read()
#             data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#             buys = data.split()[:5]
#             losses = data.split()[5:10]
#             for buy in buys:
#                 tempBuy.append(float(buy))
#             for loss in losses:
#                 tempLoss.append(float(loss))
    
#         if path == "C2":
#             with open(os.path.join(os.getcwd(), 'tuned_optimal_buys/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
#                 tempBuy = []
#                 loss2 = []
#                 data = file.read()
#                 data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
#                 buys = data.split()[:5]
#                 losses = data.split()[5:10]
#                 for buy in buys:
#                     tempBuy.append(float(buy))
#                 for loss in losses:
#                     loss2.append(float(loss))
#             tempLoss[0] = loss2[0]
#             tempLoss[2] = loss2[2]
#     if path == "B2":
#         print("TEMPLOSS2e2", tempLoss)
#     tunedbuy2e2.append(tempBuy)
#     tunedloss2e2.append(tempLoss)

for ind, path in enumerate(paths):
    if path == "A2":
        a = []
        with open(os.path.join(os.getcwd(), 'tuned_optimal_buys/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
            tempBuy = []
            tempLoss = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            buys = data.split()[:5]
            losses = data.split()[5:10]
            for buy in buys:
                tempBuy.append(float(buy))
            for loss in losses:
                a.append(float(loss))
        with open(os.path.join(os.getcwd(), 'tuned_buys_loss/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
            tempBuy = []
            tempLoss = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            buys = data.split()[:5]
            losses = data.split()[5:10]
            for buy in buys:
                tempBuy.append(float(buy))
            for loss in losses:
                a.append(float(loss))
        # with open(os.path.join(os.getcwd(), 'tuned_optimal/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
        #     tempBuy = []
        #     tempLoss = []
        #     data = file.read()
        #     data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
        #     buys = data.split()[:5]
        #     losses = data.split()[5:10]
        #     for buy in buys:
        #         tempBuy.append(float(buy))
        #     for loss in losses:
        #         a.append(float(loss))
        with open(os.path.join(os.getcwd(), 'tuned_buys_final/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
            tempBuy = []
            tempLoss = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            buys = data.split()[:5]
            losses = data.split()[5:10]
            for buy in buys:
                tempBuy.append(float(buy))
            for loss in losses:
                a.append(float(loss))
        print("AASDNJKAS", a)

    if select2e2[ind] == "ob":
        print("HELLO")
        with open(os.path.join(os.getcwd(), 'tuned_optimal_buys/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
            tempBuy = []
            tempLoss = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            buys = data.split()[:5]
            losses = data.split()[5:10]
            for buy in buys:
                tempBuy.append(float(buy))
            for loss in losses:
                tempLoss.append(float(loss))
    elif select2e2[ind] == "l":
        print("LOSS")
        with open(os.path.join(os.getcwd(), 'tuned_buys_loss/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
            tempBuy = []
            tempLoss = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            buys = data.split()[:5]
            losses = data.split()[5:10]
            for buy in buys:
                tempBuy.append(float(buy))
            for loss in losses:
                tempLoss.append(float(loss))
    elif select2e2[ind] == "o":
        print("HERE")
        with open(os.path.join(os.getcwd(), 'tuned_optimal/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
            tempBuy = []
            tempLoss = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            buys = data.split()[:5]
            losses = data.split()[5:10]
            for buy in buys:
                tempBuy.append(float(buy))
            for loss in losses:
                tempLoss.append(float(loss))
    else:
        with open(os.path.join(os.getcwd(), 'tuned_buys_final/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
            tempBuy = []
            tempLoss = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            buys = data.split()[:5]
            losses = data.split()[5:10]
            for buy in buys:
                tempBuy.append(float(buy))
            for loss in losses:
                tempLoss.append(float(loss))
    
        if path == "C2":
            with open(os.path.join(os.getcwd(), 'tuned_optimal_buys/2e2_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
                tempBuy = []
                loss2 = []
                data = file.read()
                data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
                buys = data.split()[:5]
                losses = data.split()[5:10]
                for buy in buys:
                    tempBuy.append(float(buy))
                for loss in losses:
                    loss2.append(float(loss))
            tempLoss[0] = loss2[0]
            tempLoss[2] = loss2[2]
    if path == "B2":
        print("TEMPLOSS2e2", tempLoss)
    tunedbuy2e2.append(tempBuy)
    tunedloss2e2.append(tempLoss)
    tunedbuy2e2std.append(np.nanstd(tempBuy))
    tunedloss2e2std.append(np.nanstd(tempLoss))
    print("2e2std", tunedloss2e2)

for ind, path in enumerate(paths):
    if select6e3[ind] == "ob":
        with open(os.path.join(os.getcwd(), 'tuned_optimal_buys/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
            tempBuy = []
            tempLoss = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            buys = data.split()[:5]
            losses = data.split()[5:10]
            for buy in buys:
                tempBuy.append(float(buy))
            for loss in losses:
                tempLoss.append(float(loss))
    elif select6e3[ind] == "l":
        with open(os.path.join(os.getcwd(), 'tuned_buys_loss/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
            tempBuy = []
            tempLoss = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            buys = data.split()[:5]
            losses = data.split()[5:10]
            for buy in buys:
                tempBuy.append(float(buy))
            for loss in losses:
                tempLoss.append(float(loss))
    else:
        with open(os.path.join(os.getcwd(), 'tuned_buys_final/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
            tempBuy = []
            tempLoss = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            buys = data.split()[:5]
            losses = data.split()[5:10]
            for buy in buys:
                tempBuy.append(float(buy))
            for loss in losses:
                tempLoss.append(float(loss))
        if path == "B2":
            with open(os.path.join(os.getcwd(), 'tuned_buys_loss/6e3_spoofer', '{}/graphs'.format(path), 'check_position_2.txt'), 'r') as file:
                loss2 = []
                data = file.read()
                data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
                buys = data.split()[:5]
                losses = data.split()[5:10]
                for loss in losses:
                    loss2.append(float(loss))
            tempLoss[3] = loss2[4]
            print("TEMPLOSS6e3", tempLoss)
    tunedbuy6e3.append(tempBuy)
    tunedloss6e3.append(tempLoss)
    tunedbuy6e3std.append(np.nanstd(tempBuy))
    tunedloss6e3std.append(np.nanstd(tempLoss))
    print("6e3std", tunedloss6e3)


# buyTunedDiff = np.subtract(tunedbuy6e3, tunedbuy2e2)
# buyStd = np.sqrt(np.divide(np.array(np.array(tunedbuy2e2std) ** 2), 40000)) 
# + np.divide(np.array(tunedbuy6e3std) ** 2, 80000))

lossTunedDiff = np.subtract(tunedloss6e3, tunedloss2e2)
print("tunedidff", lossTunedDiff)
lossTunedStd = np.sqrt(np.divide(np.array(tunedloss2e2std) ** 2, 40000) + np.divide(np.array(tunedloss6e3std) ** 2, 40000))
print("AD", lossTunedStd)
# print("VALUES", buyTunedDiff, lossTunedDiff)
# print("DIFFS", buyStd, lossTunedStd)
# print("UNITS", buyTunedDiff)
# sells2e2 = np.subtract(np.array(tunedbuy2e2)[:,-1], positionFast)
# sells6e3 = np.subtract(np.array(tunedbuy6e3)[:,-1], positionSlow)
# print("DIFFSELLS", np.subtract(sells2e2,sells6e3))
# print("LOSS", lossTunedDiff)

a = np.mean(lossTunedDiff, axis=1)
# print("LOSS", a)

# print("SURPLUSDIFFSELLS")
# loss2e2 = np.subtract(valueFast, np.array(tunedloss2e2)[:,-1])
# loss6e3 = np.subtract(valueSlow, np.array(tunedloss6e3)[:,-1])
# print("ASDASDNSJKADN", np.divide(np.subtract(valueFast, valueSlow), valueSlow))

# print("POSITION DIFF", np.subtract(positionSlow, positionFast))


fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
fig2, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True)

# width = 0.3

color_set1 = VALUECOMPARE[0]
color_set2 = VALUECOMPARE[1]
color_set3 = VALS[0]

for ind,i in enumerate(lossTunedDiff):
    # Plotting the bars
    # bar1 = ax.bar(ind, np.mean(i), width=width, label="Static", color=color_set1)
    # bar2 = ax.bar(ind, np.mean(buyRLDiff[ind]), width=width, label="RL", color=color_set2)
    # print(i[-1])
    # print(a[ind])
    # bar2 = ax.bar(ind, i[-1], label='Tuned', yerr=buyStd[ind], capsize=2, error_kw=dict(lw=1, capthick=1), color=color_set3)

    # bar3 = ax2.bar(ind, np.mean(lossStaticDiff[ind]), width=width, label="Static", color=color_set1)
    # bar4 = ax2.bar(ind, np.mean(lossRLDiff[ind]) ,width=width, label='RL' , color=color_set2)
    bar5 = ax2.bar(ind, a[ind], label='Tuned', yerr=lossTunedStd[ind], capsize=2, error_kw=dict(lw=1, capthick=1), color=color_set3)

# ax.set_xlabel('MM Configuration', fontsize=27,  color='black')
# ax.set_ylabel('Average Buy Difference', fontsize=27,  color='black')
ax2.set_xlabel('MM Configuration', fontsize=27,  color='black')
ax2.set_ylabel('Average Difference in Loss', fontsize=27,  color='black')
# ax.set_title('ZI Agent Baseline Surplus Difference')
# ax.set_xticks(np.arange(len(pathNames)))
ax2.set_xticks(np.arange(len(pathNames)))
# handles, labels = ax.get_legend_handles_labels()
# ax.tick_params(axis='y', labelsize=25, labelcolor='black', width=2)
ax2.tick_params(axis='y', labelsize=25, labelcolor='black', width=2)
# ax.legend(handles[:2], labels[:2], prop={'size': 15})
# ax2.legend(handles[:2], labels[:2], prop={'size': 15})
# ax.set_xticklabels(pathNames, fontsize=25,  color='black')
ax2.set_xticklabels(pathNames, fontsize=25,  color='black')
# fig.savefig(os.getcwd() + "/new_data_compil/combined" + "/buys.pdf")
fig2.savefig(os.getcwd() + "/new_data_compil/combined" + "/loss10.pdf")


# fig, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True)
# for ind, i in enumerate(baselineVals):
#     static = np.subtract(staticVals[ind], baselineVals[ind])
#     rl = np.subtract(rlVals[ind], baselineVals[ind])
#     tuned = np.subtract(tunedVals[ind], baselineVals[ind])

#     barHBL = ax2.bar(ind - width, static[1],   width=width, label='Static', color=color_set1)
#     bar2HBL = ax2.bar(ind, rl[1], width=width,label='RL', color=color_set2)
#     bar3HBL = ax2.bar(ind + width, tuned[1],   width=width, label='Tuned', color=color_set3)

# ax2.set_xlabel('Market Configuration', fontsize=27,  color='black')
# ax2.set_ylabel('HBL Surplus - Baseline', fontsize=27,  color='black')
# ax2.tick_params(axis='y', labelsize=25, labelcolor='black', width=2)
# # ax2.set_title('HBL Agent Baseline Surplus Difference')
# ax2.set_xticks(np.arange(len(pathNames)))
# handles, labels = ax2.get_legend_handles_labels()
# ax2.legend(handles[:3], labels[:3], prop={'size': 15})
# ax2.set_xticklabels(pathNames, fontsize=25,  color='black')

# plt.savefig(os.getcwd() + "/new_data_compil/combined/2e2_spoofer" + "/hbl.pdf")

# fig, ax3 = plt.subplots(figsize=(8, 6), constrained_layout=True)
# for ind, i in enumerate(baselineVals):
#     static = np.subtract(staticVals[ind], baselineVals[ind])
#     rl = np.subtract(rlVals[ind], baselineVals[ind])
#     tuned = np.subtract(tunedVals[ind], baselineVals[ind])
#     barSpoof = ax3.bar(ind - width, static[3],   width=width, label='Static', color=color_set1)
#     bar2Spoof = ax3.bar(ind, rl[3], width=width, label='RL', color=color_set2)
#     bar3Spoof = ax3.bar(ind + width, tuned[3],   width=width, label='Tuned', color=color_set3)


# ax3.set_xlabel('Market Configuration', fontsize=27,  color='black')
# ax3.set_ylabel('Spoofer Surplus', fontsize=27,  color='black')
# ax3.tick_params(axis='y', labelsize=25, labelcolor='black', width=2)

# handles, labels = ax3.get_legend_handles_labels()
# ax3.legend(handles[:3], labels[:3], prop={'size': 15})
# ax3.set_xticks(np.arange(len(pathNames)))
# ax3.set_xticklabels(pathNames, fontsize=25,  color='black')

# plt.savefig(os.getcwd() + "/new_data_compil/combined/2e2_spoofer" + "/spoofer.pdf")




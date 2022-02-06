import os
import matplotlib.pyplot as plt
import json
import numpy as np


def smooth(x, n):
    y = []
    for i in range(len(x)):
        u = 0
        v = 0
        for j in range(-n, n + 1):
            k = i + j
            if k >= 0 and k < len(x):
                u += x[k]
                v += 1
        y.append(u / v)
    return y


def draw(workspace_path, map_name_filter='', smooth_n=0):

    colors = [
              [0, 0, 0],
              [216, 30, 54], # red
              [160, 32, 240],
              [254, 151, 0],
              [0, 255, 0],
              [55, 126, 184],
             ]
    colors = np.array(colors) / 255
    color_i = 0

    os.chdir(workspace_path+'\\processed')
    plt.figure(figsize=(40, 10))
    figure = plt.figure(figsize=(32, 15))
    for root, dirs, files in os.walk('.'):
        if len(files) == 0 or root == '.':
            continue
        print("root=", root)
        xs = []
        ys = []
        n = -1
        for file_name in files:
            if file_name[-5:] != '.json':
                continue
            fo = open(root + '\\' + file_name, 'r')
            print(root, file_name)
            info = json.load(fo)
            x = info['test_battle_won_mean_T']
            y = info['test_battle_won_mean']
            xs.append(x)
            ys.append(y)
            if n == -1:
                n = len(x)
            n = min(n, len(x), len(y))
        for i in range(len(xs)):
            xs[i] = xs[i][:n]
            ys[i] = ys[i][:n]
        xs = np.array(xs)
        ys = np.array(ys)

        print(xs.shape)

        print(np.array(ys))

        x = np.mean(xs, axis=0).tolist()
        y = np.mean(ys, axis=0).tolist()
        low = np.percentile(ys, 25, axis=0).tolist()
        high = np.percentile(ys, 75, axis=0).tolist()

        y = smooth(y, smooth_n)
        low = smooth(low, smooth_n)
        high = smooth(high, smooth_n)

        label = root[2:]
        # plt.plot(x, y, label=label, linestyle='-', linewidth=4, c=colors[color_i])
        # plt.fill_between(x, low, high, alpha=0.1, facecolor=colors[color_i])
        plt.plot(x, y, label=label, linestyle='-', linewidth=4)
        plt.fill_between(x, low, high, alpha=0.1)
        color_i += 1
    # plt.xlim(0, 2.5e5)
    plt.xlabel(map_name_filter + ' T_env', fontsize=52)
    plt.ylabel('Win%', fontsize=52)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=16)
    figure.tight_layout()
    figure.savefig(workspace_path+'//t_env.png')


workspace_path = 'D:\\THU\\RL\\adversarial_MARL\\results\\10m_vs_11m'
draw(workspace_path, map_name_filter='10m_vs_11m', smooth_n=7)

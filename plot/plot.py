from locale import dcgettext
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
        
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

def draw(ax, path, tagdata, x_lim=1, ylabel='Win%', smooth_n=7):
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

    title = path[17:].capitalize()

    for algo in tagdata.keys():
        xs = tagdata[algo][0]
        ys = tagdata[algo][1]
 
        if title.startswith('Pursuit'):
            ys = 5 - ys
            ylabel = ylabel.replace("left", "caught")
        xs = xs / (1e6 + 0.0)

        x = np.mean(xs, axis=0).tolist()
        y = np.mean(ys, axis=0).tolist()

        low = np.percentile(ys, 25, axis=0).tolist()
        high = np.percentile(ys, 75, axis=0).tolist()

        y = smooth(y, smooth_n)
        low = smooth(low, smooth_n)
        high = smooth(high, smooth_n)

        label = algo

        y[0] = 0
        low[0] = 0
        high[0] = 0
        
        ax.plot(x, y, label=label, linestyle='-', linewidth=4)
        ax.fill_between(x, low, high, alpha=0.1)
        color_i += 1

    if title.startswith('Pursuit'):
        x,y,low,high = read_dicg_pursuit()
        y = smooth(y, smooth_n)
        low = smooth(low, smooth_n)
        high = smooth(high, smooth_n)
        ax.plot(x, y, label='DICG', linestyle='-', linewidth=4)
        ax.fill_between(x, low, high, alpha=0.1)
        color_i += 1
        title = 'Pursuit'

    ax.set_title(title, fontsize=30)
    ax.grid()
    ax.set_xlim(0, x_lim)
    ax.set_xlabel('T(mil)', fontsize=30)
    ax.set_ylabel(ylabel.replace('_',' ').capitalize(), fontsize=30)
    ax.tick_params(axis='x',labelsize=20)
    ax.tick_params(axis='y',labelsize=20)

def read_dicg_pursuit():
    dicg1 = np.load('../result/result.pursuit-dicg/1.npy')
    dicg2 = np.load('../result/result.pursuit-dicg/2.npy')
    dicg3 = np.load('../result/result.pursuit-dicg/3.npy')
    xs = np.array([dicg1[:,0], dicg2[:,0], dicg3[:,0]])

    xs = xs / (1e6 + 0.0)
    x = np.mean(xs, axis=0).tolist()

    ys = np.array([dicg1[:,1], dicg2[:,1], dicg3[:,1]])
    ys = 5 - ys
    y = np.mean(ys, axis=0).tolist()
    low = np.percentile(ys, 25, axis=0).tolist()
    high = np.percentile(ys, 75, axis=0).tolist()

    return x,y,low,high

def read_dicg_mmm2():
    dicg1 = np.load('../../temp_draw/mmm2-dicg/1.npy')
    dicg2 = np.load('../../temp_draw/mmm2-dicg/2.npy')
    xs = np.array([dicg1[:,0], dicg2[:,0]])

    xs = xs / (1e6 + 0.0)
    
    x = np.mean(xs, axis=0).tolist()

    ys = np.array([dicg1[:,1], dicg2[:,1]])
    #ys = 5 - ys
    y = np.mean(ys, axis=0).tolist()
    low = np.percentile(ys, 25, axis=0).tolist()
    high = np.percentile(ys, 75, axis=0).tolist()

    return x,y,low,high


def getdata_json(path, labels):
    dir2label = {
        '../result/result.aloha': 
            {
                'cgmix_mixemb2_alpha0.5_onff4': 'NL-CG (embed=2,enum)',
                'cgmix_mixemb3_alpha0.5_onff8': 'NL-CG (embed=3,enum)',
                'cgmix_mixemb4_alpha0.5_onff8': 'NL-CG (embed=4,iterative $n_{max}$=8)',
                'cgmix_mixemb4_alpha0.5_onff16': 'NL-CG (embed=4,enum)',
                'cgmix_mixemb10_alpha0.5_onff16': 'NL-CG (embed=10,iterative $n_{max}$=16)',
                'cgmix_mixemb10_alpha0.5_onff128': 'NL-CG (embed=10,enum)',
                'dcg_duelling': 'DCG',
                'qmix_mixemb32': 'QMIX (embed=32)',
            },
        '../result/result.sensor': 
            {
                'cgmix_mixemb2_alpha0.5_onff4': 'NL-CG (embed=2,enum)',
                'cgmix_mixemb3_alpha0.5_onff8': 'NL-CG (embed=3,enum)',
                'cgmix_mixemb4_alpha0.5_onff8': 'NL-CG (embed=4,iterative $n_{max}$=8)',
                'cgmix_mixemb4_alpha0.5_onff16': 'NL-CG (embed=4,enum)',
                'cgmix_mixemb10_alpha0.5_onff16': 'NL-CG (embed=10,iterative $n_{max}$=16)',
                'cgmix_mixemb10_alpha0.5_onff128': 'NL-CG (embed=10,enum)',
                'dcg_duelling': 'DCG',
                'qmix_mixemb32': 'QMIX (embed=32)',
            },
        '../result/result.gather': 
            {
                'cgmix_mixemb2_alpha0.5_onff4': 'NL-CG (embed=2,enum)',
                'cgmix_mixemb3_alpha0.5_onff8': 'NL-CG (embed=3,enum)',
                'cgmix_mixemb4_alpha0.5_onff8': 'NL-CG (embed=4,iterative $n_{max}$=8)',
                'cgmix_mixemb4_alpha0.5_onff16': 'NL-CG (embed=4,enum)',
                'cgmix_mixemb10_alpha0.5_onff16': 'NL-CG (embed=10,iterative $n_{max}$=16)',
                'cgmix_mixemb10_alpha0.5_onff128': 'NL-CG (embed=10,enum)',
                'dcg_duelling': 'DCG',
                'qmix_mixemb32': 'QMIX (embed=32)',
            },
        '../result/result.hallway': 
            {
                'cgmix_mixemb2_alpha0.5_onff4': 'NL-CG (embed=2,enum)',
                'cgmix_mixemb3_alpha0.5_onff8': 'NL-CG (embed=3,enum)',
                'cgmix_mixemb4_alpha0.5_onff16': 'NL-CG (embed=4,enum)',
                'cgmix_mixemb10_alpha0.5_onff128': 'NL-CG (embed=10,enum)',
            },
        '../result/result.pursuit': 
            {
                'cgmix_mixemb3_alpha0.5_onff8': 'NL-CG (embed=3,enum)',
                'cgmix_mixemb4_alpha0.5_onff8': 'NL-CG (embed=4,iter)',
                'cgmix_mixemb16_alpha0.5_onff32': 'NL-CG (embed=16,iter)',
                'dcg_duelling': 'DCG',
                'qmix_mixemb32': 'QMIX (embed=32)',
            },
    }
    algo_infos = {}
    for algo in os.listdir(path):
        if algo not in dir2label[path]:
            continue
        label = dir2label[path][algo]
        if label not in labels:
            continue
        seed_infos = []
        algo_path = os.path.join(path, algo, 'sacred')
        for seed in os.listdir(algo_path):
            if seed.startswith('_'):
                continue
            seed_path = os.path.join(algo_path, seed)
            seed_info_file = os.path.join(seed_path, 'info.json')
            #print(seed_info_file)
            with open(seed_info_file, 'r') as f:
                json_data = json.load(f)
            seed_infos.append(json_data)
        if seed_infos:
            algo_infos[label] = seed_infos
    assert len(algo_infos) == len(labels)
    return algo_infos

def getdata4tag_real(algo_infos, tag, least_len=None):
    y_tag = tag
    x_tag = tag + '_T'
    num_algos = len(algo_infos)

    tagdata = {}
    for algo in algo_infos.keys():
        seed_infos = algo_infos[algo]
        num_seeds = len(seed_infos)
        
        tagdata4algo_x = []
        tagdata4algo_y = []
        
        algo_least_len = np.inf

        for ns in range(num_seeds):
            x = seed_infos[ns][x_tag]
            y = seed_infos[ns][y_tag]

            assert len(x) == len(y)

            if len(x) < algo_least_len:
                algo_least_len = len(x)

            if least_len:
                dx = x[-1] - x[-2]
                for d in range(len(x), least_len):
                    x.append(x[-1] + dx)
                    y.append(y[-1])

            tagdata4algo_x.append(x)
            tagdata4algo_y.append(y)


        if least_len and algo_least_len < least_len:
            algo_least_len = least_len

        for ns in range(num_seeds):
            tagdata4algo_x[ns] = tagdata4algo_x[ns][:algo_least_len]
            tagdata4algo_y[ns] = tagdata4algo_y[ns][:algo_least_len]
        tagdata[algo] = [np.array(tagdata4algo_x), np.array(tagdata4algo_y)]
    return tagdata

def draw_fig(figno=1):
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))
    axes = axes.flatten()

    if figno == 1:
        #FIGURE 1    
        paths = ['../result/result.aloha', '../result/result.sensor', '../result/result.gather',]
        tags = ['test_trans_mean', 'test_scaned_mean', 'test_battle_won_mean', ]
        x_lims = [1.5, 0.2, 1.5]
        labels = ['NL-CG (embed=3,enum)',  'NL-CG (embed=4,iterative $n_{max}$=8)', 'DCG',  'QMIX (embed=32)']
    elif figno == 2:    
        #FIGURE2
        paths = ['../result/result.aloha', '../result/result.sensor', '../result/result.hallway']
        tags = ['test_trans_mean', 'test_scaned_mean', 'test_battle_won_mean']
        x_lims = [1.5, 0.2, 2]
        labels = ['NL-CG (embed=2,enum)', 'NL-CG (embed=3,enum)',  'NL-CG (embed=4,enum)', 'NL-CG (embed=10,enum)',]
    elif figno == 3:
        # #FIGURE3
        paths = ['../result/result.aloha', '../result/result.sensor', '../result/result.gather']
        tags = ['test_trans_mean', 'test_scaned_mean', 'test_battle_won_mean']
        x_lims = [1.5, 0.2, 1.5]
        labels = ['NL-CG (embed=4,enum)', 'NL-CG (embed=10,iterative $n_{max}$=16)']
    for axno in range(3):
        algo_infos = getdata_json(paths[axno], labels)
        tagdata = getdata4tag_real(algo_infos, tags[axno])
        title = paths[axno][17:]
        draw(axes[axno], paths[axno], tagdata, x_lims[axno], tags[axno], 7)
        
    lines0, labels0 = axes[0].get_legend_handles_labels()
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = axes[2].get_legend_handles_labels()
    assert labels0 == labels1 == labels2

    #assert False ###Check labels for different axes
    fig.legend(labels0, loc="upper center", ncol=2, fontsize=40,)
    fig.tight_layout()
    fig.subplots_adjust(top=0.75, hspace=0.5)
    fig.savefig('../plot_results/perform/'+str(figno)+'cam.png')

def draw_pursuit():
    fig, axes = plt.subplots(1, 1, figsize=(11, 10))
    axes = [axes]

    paths = ['../result/result.pursuit']
    tags = ['test_prey_left_mean']
    x_lims = [2]
    labels = ['NL-CG (embed=3,enum)', 'NL-CG (embed=4,iter)', 'NL-CG (embed=16,iter)', 'DCG', 'QMIX (embed=32)']

    for axno in range(1):
        algo_infos = getdata_json(paths[axno], labels)
        tagdata = getdata4tag_real(algo_infos, tags[axno], int(x_lims[axno]*100))
        title = paths[axno][17:]
        draw(axes[axno], paths[axno], tagdata, x_lims[axno], tags[axno], 1)
        axes[axno].set_xlabel('T(mil)', fontsize=30)
        
    lines, labels = axes[0].get_legend_handles_labels()
    order = [1,2,0,4,3,5]
    fig.legend([lines[idx] for idx in order], [labels[idx] for idx in order], loc="upper center", ncol=2, fontsize=25,)
    #fig.legend(labels, loc="upper center", ncol=2, fontsize=25,)
    #fig.legend(labels, loc="upper center", ncol=2, fontsize=15,)
    fig.tight_layout()
    fig.subplots_adjust(top=0.75, hspace=0.3)
    fig.savefig('../plot_results/perform/pursuit-cam.png')


if __name__ == "__main__":
    draw_fig(1)
    draw_fig(2)
    draw_fig(3)
    draw_pursuit()
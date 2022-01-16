import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

def getdata():    
    basecond = [[18, 20, 19, 18, 13, 4, 1],                
                [20, 17, 12, 9, 3, 0, 0],               
                [20, 20, 20, 12, 5, 3, 0]]    

    cond1 = [[18, 19, 18, 19, 20, 15, 14],             
             [19, 20, 18, 16, 20, 15, 9],             
             [19, 20, 20, 20, 17, 10, 0],             
             [20, 20, 20, 20, 7, 9, 1]]   

    cond2 = [[20, 20, 20, 20, 19, 17, 4],            
             [20, 20, 20, 20, 20, 19, 7],            
             [19, 20, 20, 19, 19, 15, 2]]   

    cond3 = [[20, 20, 20, 20, 19, 17, 12],           
             [18, 20, 19, 18, 13, 4, 1],            
             [20, 19, 18, 17, 13, 2, 0],            
             [19, 18, 20, 20, 15, 6, 0]]    

    return basecond, cond1, cond2, cond3

def getdata_json():
    path = 'results.toy.new'
    algo_infos = {}
    for algo in os.listdir(path):
        seed_infos = []
        algo_path = os.path.join(path, algo, 'sacred')
        for seed in os.listdir(algo_path):
            if seed.startswith('_'):
                continue
            seed_path = os.path.join(algo_path, seed)
            seed_info_file = os.path.join(seed_path, 'info.json')
            with open(seed_info_file, 'r') as f:
                json_data = json.load(f)
            seed_infos.append(json_data)
        algo_infos[algo] = seed_infos
    return algo_infos

def getdata4tag(algo_infos, tag):
    num_algos = len(algo_infos)

    tagdata = {}
    for algo in algo_infos.keys():
        seed_infos = algo_infos[algo]
        num_seeds = len(seed_infos)
        
        tagdata4algo = []
        for ns in range(num_seeds):
            seed_tag_result = []
            for data in seed_infos[ns][tag]:
                seed_tag_result.append(data['value'])
            tagdata4algo.append(seed_tag_result)
        tagdata[algo] = tagdata4algo
    #import pdb; pdb.set_trace()
    return tagdata

tag = 'test_return_mean'
algo_infos = getdata_json()
tagdata = getdata4tag(algo_infos, tag)

linestyle = ['-', '--', ':', '-.']
color = ['r', 'g', 'b', 'k']

fig = plt.figure()

style = 0
assert len(linestyle) == len(color)
style_num = len(linestyle)
for algo in tagdata.keys():
    xdata = np.array([0, 1, 2, 3, 4, 5])
    sns.tsplot(time=xdata, data=tagdata[algo], color=color[style], linestyle=linestyle[style], condition=algo)
    style += 1
    style %= style_num

plt.ylabel(tag, fontsize=25)
plt.xlabel("Step", fontsize=25)
#plt.title("Awesome Robot Performance", fontsize=30)
plt.show()

'''
linestyle = ['-', '--', ':', '-.']
color = ['r', 'g', 'b', 'k']
label = ['algo1', 'algo2', 'algo3', 'algo4']

for i in range(4):    
    sns.tsplot(time=xdata, data=data[i], color=color[i], linestyle=linestyle[i], condition=label[i])

plt.ylabel("Success Rate", fontsize=25)
plt.xlabel("Iteration Number", fontsize=25)
plt.title("Awesome Robot Performance", fontsize=30)
plt.show()
'''
import os
import shutil
import matplotlib.pyplot as plt
import json


def create_target_files(workspace_path, map_name_filter=None, del_config=[]):
    os.chdir(workspace_path)
    if os.path.exists('processed'):
        shutil.rmtree('processed')
    os.mkdir('processed')
    os.chdir(os.path.join(workspace_path, 'to_process'))
    paths = []
    for root, dirs, files in os.walk('.'):
        for name in dirs:
            paths.append(os.path.join(workspace_path, 'to_process', root[2:], name))

    config_dict = {}

    for path in paths:
        os.chdir(path)
        if not os.path.exists("config.json"):
            continue
        fo = open("config.json", "r")
        config = json.load(fo)
        for to_del in del_config:
            if to_del in config:
                del config[to_del]

        if map_name_filter is not None:
            map_name = str(config["env_args"]["map_name"])
            if map_name != map_name_filter:
                continue

        print("config:", str(config))

        if not str(config) in config_dict:
            config_dict[str(config)] = config
    
    config_paths = {}
    for ux in config_dict:
        for vx in config_dict:
            if ux != vx:
                u = config_dict[ux]
                v = config_dict[vx]
                for x in u:
                    if not x in v:
                        config_paths[x] = 0
                    elif u[x] != v[x]:
                        config_paths[x] = 0

    CNT = 0

    for x in config_dict:
        y = ""
        for z in config_paths:
            if not z in config_dict[x]:
                continue
            if len(y) != 0:
                y += '_'
            y += str(z) + str(config_dict[x][z])
        if len(y) == 0:
            y = "result"

        CNT += 1

        while True:
            i = y.find('[')
            if i != -1:
                y = y[:i] + y[i+1:]
                continue
            i = y.find(']')
            if i != -1:
                y = y[:i] + y[i+1:]
                continue
            break

        config_dict[x] = y
        os.chdir(workspace_path+'\\processed')
        os.mkdir(y)
    
    
    for path in paths:
        os.chdir(path)

        if not os.path.exists("info.json"):
            continue

        print(os.getcwd())
        fo = open("config.json", "r")
        config = json.load(fo)
        for to_del in del_config:
            if to_del in config:
                del config[to_del]

        if map_name_filter is not None:
            map_name = str(config["env_args"]["map_name"])
            if map_name != map_name_filter:
                continue

        path_name = config_dict[str(config)]
        print('\t', path_name)
        os.chdir(workspace_path+'\\processed\\'+path_name)
        count = 1
        for root, dirs, files in os.walk('.'):
            count += len(files)
        file_name = str(count)
        target_path = workspace_path+'\\processed\\'+path_name
        target_file = target_path + '\\' + file_name
        os.chdir(path)
        shutil.copyfile("info.json", target_file+".json")
        shutil.copyfile("cout.txt", target_file+"cout.txt")


del_config = [
    'save_replay',
    't_max',
    'seed',
    'max_gradient_attack_binary_search_step',
    'gradient_attack_binary_search_step',
    'expl_mac',
    'controller_log_interval',
    'save_model_interval',
    'save_replay_interval',
    'save_replay',
    'save_model',
    'adversarial_agent_id',
    'use_theta_diff'
]
workspace_path = 'D:\\THU\\RL\\adversarial_MARL\\results\\10m_vs_11m'
create_target_files(workspace_path, map_name_filter='10m_vs_11m', del_config=del_config)

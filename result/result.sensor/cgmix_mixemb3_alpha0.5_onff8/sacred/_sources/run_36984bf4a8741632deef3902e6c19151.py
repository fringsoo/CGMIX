from copy import deepcopy
from random import random
from re import T
import numpy as np
from torch.nn.functional import normalize
import  pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter 
from utils.Heuristic import GreedyActionSelector

import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    #unique_token = _config['checkpoint_path'].split('/')[-2]
    args.unique_token = unique_token

    args.local_results_path = os.path.join(args.local_results_dir, args.local_results_path)

    if args.use_tensorboard:
        #tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results.toy", "tb_logs")
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), args.local_results_path, "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    if args.use_sacred:
        # sacred is on by default
        logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def compare(mac, batch, args, compare_onoff_configamount):
    rewards = batch["reward"][:, :-1]
    actions = batch["actions"][:, :-1]
    terminated = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

    p = normalize(batch['avail_actions'].float(),p=1,dim=3).cpu().numpy()
    random_actions = np.zeros(p.shape[:-1])
    for bb in range(p.shape[0]):
        for tt in range(p.shape[1]):
            for aa in range(p.shape[2]):
                if abs(sum(p[bb,tt,aa]) - 1) < 1e-3:
                    random_actions[bb,tt,aa] = np.random.choice(p.shape[3],1,p=p[bb,tt,aa])[0]
                else:
                    assert sum(p[bb,tt,aa]) == 0
                    random_actions[bb,tt,aa] = np.random.choice(p.shape[3],1)[0]
    random_actions = random_actions.astype(np.int)

    # Calculate estimated Q-Values
    mac_f_i = []
    mac_f_ij = []
    ts = []

    mac_f_i_compare = []
    mac_f_ij_compare = []
    ts_compare = []

    mac_f_i_dcg = []
    mac_f_ij_dcg = []

    mac_f_i_random = []
    mac_f_ij_random = []

    args_compare = deepcopy(args)
    args_compare.onoff_configamount = compare_onoff_configamount
    
    greedy_action_selector_compare = GreedyActionSelector(args_compare)

    mac.init_hidden(batch.batch_size)
    for t in range(batch.max_seq_length - 1):
        f_i, f_ij = mac.annotations(batch, t)
        state = batch["state"][:, t]
        w_1, w_final, bias = mac.mixer.get_para(state)
        avail_actions = batch['avail_actions'][:, t]

        def select_process():
            if os.path.exists('src/utils/__pycache__'): import shutil; shutil.rmtree('src/utils/__pycache__')  
            greedy, t_run, wholeiterttime, maxsumtime, maxsumiter = mac.greedy_action_selector.solve_graph(f_i, f_ij, w_1, w_final, bias, avail_actions=avail_actions, device=f_i.device)
            q_i, q_ij = mac.q_values(f_i, f_ij, greedy)
            mac_f_i.append(q_i)
            mac_f_ij.append(q_ij)
            ts.append(maxsumtime*1000)

        def select_process_compare():
            if os.path.exists('src/utils/__pycache__'): import shutil; shutil.rmtree('src/utils/__pycache__')  
            greedy_compare, t_compare, wholeiterttime_compare, maxsumtime_compare, maxsumiter_compare = greedy_action_selector_compare.solve_graph(f_i, f_ij, w_1, w_final, bias, avail_actions=avail_actions, device=f_i.device)
            q_i_compare, q_ij_compare = mac.q_values(f_i, f_ij, greedy_compare)
            mac_f_i_compare.append(q_i_compare)
            mac_f_ij_compare.append(q_ij_compare)
            ts_compare.append(maxsumtime_compare*1000)

        if np.random.rand(1) == 0:
            select_process()
            select_process_compare()
        else:
            select_process_compare()
            select_process()


        dcg_action = mac.max_sum_C_graph(f_i, f_ij, avail_actions)
        q_i_dcg, q_ij_dcg = mac.q_values(f_i, f_ij, dcg_action)
        mac_f_i_dcg.append(q_i_dcg)
        mac_f_ij_dcg.append(q_ij_dcg)
        
        random_action = th.from_numpy(random_actions[:,t,:]).cuda()[:,:,None]
        q_i_random, q_ij_random = mac.q_values(f_i, f_ij, random_action)
        mac_f_i_random.append(q_i_random)
        mac_f_ij_random.append(q_ij_random)

        
        #assert th.all(greedy == actions[:,t,:,:])      
        #assert th.all(greedy == greedy_compare)   
        # print(actions[:,t,:,:], greedy, greedy_compare)

    mac_f_i = th.stack(mac_f_i, dim=1)
    mac_f_ij = th.stack(mac_f_ij, dim=1)
    mac_out = th.cat((mac_f_i, mac_f_ij), dim=2)
    
    mac_f_i_compare = th.stack(mac_f_i_compare, dim=1)
    mac_f_ij_compare = th.stack(mac_f_ij_compare, dim=1)
    mac_out_compare = th.cat((mac_f_i_compare, mac_f_ij_compare), dim=2)

    mac_f_i_dcg = th.stack(mac_f_i_dcg, dim=1)
    mac_f_ij_dcg = th.stack(mac_f_ij_dcg, dim=1)
    mac_out_dcg = th.cat((mac_f_i_dcg, mac_f_ij_dcg), dim=2)

    mac_f_i_random = th.stack(mac_f_i_random, dim=1)
    mac_f_ij_random = th.stack(mac_f_ij_random, dim=1)
    mac_out_random = th.cat((mac_f_i_random, mac_f_ij_random), dim=2)

    # Mix
    chosen_action_qval = mac.mixer(mac_out, batch["state"][:, :-1]) ###1 or 2 or -1!!!
    chosen_action_qval_compare = mac.mixer(mac_out_compare, batch["state"][:, :-1])
    chosen_action_qval_dcg = mac.mixer(mac_out_dcg, batch["state"][:, :-1])
    chosen_action_qval_random = mac.mixer(mac_out_random, batch["state"][:, :-1])
    
    assert th.all(chosen_action_qval <= chosen_action_qval_compare)

    assert batch["reward"].shape[0]==1 and batch["reward"].shape[1]==batch.max_seq_length and batch["reward"].shape[2]==1 and len(batch["reward"].shape)==3
    batchrewards = batch["reward"].cpu().numpy()
    qvalue_groundtruth = np.zeros(batchrewards.shape)
    qvalue_groundtruth[:,-1,:] = batchrewards[:,-1,:]
    for tt in range(batchrewards.shape[1]-2,-1,-1):
        qvalue_groundtruth[:,tt,:] = batchrewards[:,tt,:] + args.gamma * qvalue_groundtruth[:,tt + 1,:]
    qvalue_groundtruth = qvalue_groundtruth[:,:-1,:]
    
    return qvalue_groundtruth, chosen_action_qval, chosen_action_qval_compare, chosen_action_qval_dcg, chosen_action_qval_random, ts, ts_compare

def evaluate_sequential_compare(args, runner):
    # args.onoff_configamount = 4
    # from utils.Heuristic import GreedyActionSelector
    # runner.mac.greedy_action_selector = GreedyActionSelector(args)

    envsavepath = args.envsavepath.capitalize()
    compare_onoff_configamount = 8
    #compare_onoff_configamount_show = 'Iterative optimization\n(till local convergence)'
    compare_onoff_configamount_show = r"Iterative optimization ($n_{max}=4$)"


    gtqvals = []
    gtqvals_dcg =[]

    qvals = []
    qvals_compare = []
    qvals_dcg = []
    qvals_random = []
    
    times = []
    times_compare = []
    
    for _ in range(100):    
        print('_______________________')
        print(_)

        trial_num = 3
        t_assigned = _% 4
        if t_assigned == 0:
            t_assigned = np.random.choice(4)
        if t_assigned == 0:
            t_assigned = np.random.choice(4)

        batches = runner.run_new2(test_mode=True, trial_num = trial_num, t_assigned=t_assigned)
        
        gtqval = []
        gtqval_dcg = []

        qval = []
        qval_compare = []
        qval_dcg = []
        qval_random = []

        bb = 0
        for batch in batches:
            seqgtqval, seqqval, seqqval_compare, seqqval_dcg, seqqval_random, ts, ts_compare = compare(runner.mac, batch, args, compare_onoff_configamount)
            
            seqgtqval = seqgtqval.squeeze().tolist()
            seqqval = seqqval.squeeze().cpu().detach().numpy().tolist()
            seqqval_compare = seqqval_compare.squeeze().cpu().detach().numpy().tolist()
            seqqval_dcg = seqqval_dcg.squeeze().cpu().detach().numpy().tolist()
            seqqval_random = seqqval_random.squeeze().cpu().detach().numpy().tolist()

            #print(seqgtqval[t_assigned], seqqval[t_assigned], seqqval_compare[t_assigned], seqqval_dcg[t_assigned], seqqval_random[t_assigned])

            if bb < trial_num:
                times += ts
                times_compare += ts_compare

                gtqval.append(seqgtqval[t_assigned])
                #qval0.append(qval[t_assigned]-0.5*np.random.binomial(1, 0.05, 1)[0])
                qval.append(seqqval[t_assigned])
                qval_compare.append(seqqval_compare[t_assigned])
                qval_dcg.append(seqqval_dcg[t_assigned])
                qval_random.append(seqqval_random[t_assigned])
            
            else:
                gtqval_dcg.append(seqgtqval[t_assigned])
            bb+=1

        gtqvals += [np.mean(gtqval)]
        gtqvals_dcg += [np.mean(gtqval_dcg)]

        qvals += [np.mean(qval)]
        qvals_compare += [np.mean(qval_compare)]
        qvals_dcg += [np.mean(qval_dcg)]
        qvals_random += [np.mean(qval_random)]
            

    ###1###
    plt.rcParams['font.family'] = ['Arial']
    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(gtqvals, qvals, edgecolor=None, c='y', s=50, marker='s')
    ax.plot(np.arange(-1e3,1e3,1),np.arange(-1e3,1e3,1),'b--', label='Line y=x',zorder=2)#红色虚线
    fontdict1 = {"size":15,"color":"k",}
    ax.set_xlabel("Ground Truth $Q_{tot}$ Values", fontdict=fontdict1)
    ax.set_ylabel("Estimated $Q_{tot}$", fontdict=fontdict1)

    ax.grid(which='major',axis='y',ls='--',c='k',alpha=.7)
    ax.set_axisbelow(True)
    
    if envsavepath == 'Aloha':
        ax.set_xlim((2, 5))
        ax.set_ylim((2, 5))
        #ax.set_xscale("symlog")
        #ax.set_yscale("symlog",basey=10)
    
    elif envsavepath == 'Sensor':
        ax.set_xlim((20, 30))
        ax.set_ylim((5, 30))
        #ax.set_xscale("symlog")
        #ax.set_yscale("symlog",basey=10)
        #plt.ticklabel_format(style='scientific',axis='y')

    elif envsavepath == 'Gather':
        ax.set_xlim((8, 11))
        ax.set_ylim((0, 11))
        #ax.set_xscale("symlog",basex=100)
        #ax.set_yscale("symlog",basey=100)

    if envsavepath == 'Sensor':
        #plt.legend(fontsize=15, loc="upper left")
        plt.legend(fontsize=15, loc="best")

    plt.legend(fontsize=15, loc="best")

    titlefontdict = {"size":18,"color":"k",}
    ax.set_title(r"NL-CG (embed=3, iterative, $n_{max}=4$)(" + envsavepath + ")",titlefontdict,pad=20)
    fig.subplots_adjust(left=0.15)
    fig.savefig('plot_results/opt_eff/' + envsavepath + '-qgt_qtot.png')
    

    

    
    ###2###
    plt.rcParams['font.family'] = ['Arial']
    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(qvals_compare, qvals_dcg, label='DCG actions', edgecolor=None, c='g', s=10,marker='s')
    #ax.scatter(qvals_compare, qvals_random, label='Random actions', edgecolor=None, c='r', s=10,marker='s')
    ax.scatter(qvals_compare, qvals, label=compare_onoff_configamount_show, edgecolor=None, c='k', s=50,marker='s')
    ax.plot(np.arange(-1e3,1e3,1),np.arange(-1e3,1e3,1),'b--', label='Line y=x',zorder=2)#红色虚线
    fontdict1 = {"size":15,"color":"k",}
    ax.set_xlabel("NL-CG (embed=3, enum)", fontdict=fontdict1)
    ax.set_ylabel("Other search strategies", fontdict=fontdict1)

    ax.grid(which='major',axis='y',ls='--',c='k',alpha=.7)
    ax.set_axisbelow(True)
    
    if envsavepath == 'Aloha':
        ax.set_xlim((2, 5))
        ax.set_ylim((2, 5))
        #ax.set_xlim((0, 10))
        #ax.set_ylim((-100, 6))
        #ax.set_xscale("symlog",basex=10, linscale=1)
        #ax.set_yscale("symlog",basey=10, linthresh=10)
    
    elif envsavepath == 'Sensor':
        ax.set_xlim((20, 30))
        ax.set_ylim((5, 30))
        #ax.set_xscale("symlog")
        #ax.set_yscale("symlog",basey=10)
        #plt.ticklabel_format(style='scientific',axis='y')

    elif envsavepath == 'Gather':
        ax.set_xlim((8, 11))
        ax.set_ylim((0, 11))
        #ax.set_xscale("symlog",basex=100)
        #ax.set_yscale("symlog",basey=100)

    if envsavepath == 'Sensor':
        #plt.legend(fontsize=15, loc="upper left")
        plt.legend(fontsize=15, loc="best")

    plt.legend(fontsize=15, loc="best")

    titlefontdict = {"size":20,"color":"k",}
    ax.set_title("$Q_{tot}$ of Found Solutions (" + envsavepath + ")",titlefontdict,pad=20)
    fig.subplots_adjust(left=0.15)
    fig.savefig('plot_results/opt_eff/' + envsavepath + '-qtot_diffselection.png')
    


    ###3###
    plt.rcParams['font.family'] = ['Arial']
    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(gtqvals, gtqvals_dcg,  edgecolor=None, c='g', s=10,marker='s')
    ax.plot(np.arange(-1e3,1e3,1),np.arange(-1e3,1e3,1),'b--', label='Line y=x',zorder=2)#红色虚线
    fontdict1 = {"size":15,"color":"k",}
    ax.set_xlabel("NL-CG (embed=3, iterative, $n_{max}=4$)", fontdict=fontdict1)
    ax.set_ylabel("DCG", fontdict=fontdict1)

    ax.grid(which='major',axis='y',ls='--',c='k',alpha=.7)
    ax.set_axisbelow(True)
    
    if envsavepath == 'Aloha':
        ax.set_xlim((2, 5))
        ax.set_ylim((-1, 5))
        #ax.set_xscale("symlog")
        #ax.set_yscale("symlog",basey=10)
    
    elif envsavepath == 'Sensor':
        ax.set_xlim((20, 30))
        ax.set_ylim((5, 30))
        #ax.set_xscale("symlog")
        #ax.set_yscale("symlog",basey=10)
        #plt.ticklabel_format(style='scientific',axis='y')

    elif envsavepath == 'Gather':
        ax.set_xlim((8, 11))
        ax.set_ylim((0, 11))
        #ax.set_xscale("symlog",basex=100)
        #ax.set_yscale("symlog",basey=100)

    if envsavepath == 'Sensor':
        #plt.legend(fontsize=15, loc="upper left")
        plt.legend(fontsize=15, loc="best")

    plt.legend(fontsize=15, loc="best")

    titlefontdict = {"size":20,"color":"k",}
    ax.set_title("Ground Truth $Q_{tot}$ Values (" + envsavepath + ")",titlefontdict,pad=20)
    fig.subplots_adjust(left=0.15)
    fig.savefig('plot_results/opt_eff/' + envsavepath + '-qgt.png')
    
    #=============================

    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.plot(np.arange(0,5,0.1),np.arange(0,5,0.1),'b--', label='Line y=x')#红色虚线
    ax.scatter(times_compare, times, label=compare_onoff_configamount_show, edgecolor=None, c='k', s=10,marker='s')
        
    fontdict1 = {"size":15,"color":"k",}
    #ax.set_xlabel("Enumerating all pieces (μs)", fontdict=fontdict1)
    ax.set_xlabel("Enumerating all pieces (ms)", fontdict=fontdict1)
    #ax.set_ylabel("Iterative optimization strategies (μs)", fontdict=fontdict1)
    ax.set_ylabel("Iterative optimization strategies (ms)", fontdict=fontdict1)
    ax.grid(which='major',axis='y',ls='--',c='k',alpha=.7)
    ax.set_axisbelow(True)

    if envsavepath == 'Aloha':
        ax.set_xlim((0, 0.4))
        ax.set_ylim((0, 0.4))
    elif envsavepath == 'Sensor':
        ax.set_xlim((0, 4))
        ax.set_ylim((0, 4))
    elif envsavepath == 'Gather':
        ax.set_xlim((0, 0.8))
        ax.set_ylim((0, 0.8))

    if envsavepath == 'Sensor':
        plt.legend(fontsize=15, loc="upper left")
    titlefontdict = {"size":15,"color":"k",}
    ax.set_title('Seaching Time Comparison ('+ envsavepath +')' ,titlefontdict,pad=20)
    fig.savefig('plot_results/opt_eff/'+ envsavepath + '-time-m4.png')

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        args.checkpoint_path = os.path.join(args.local_results_path, "models", args.checkpoint_path)

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential_compare(args, runner)
            #evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config

import torch as th
import numpy as np
from utils.c_utils import load_c_lib, c_ptr, c_int, c_longlong, c_float, c_double
import copy
import ctypes
import time

def preprocess_values(f, _g, avail_actions):
    n_agents, n_actions = f.shape[1], f.shape[2]
    g = th.zeros((f.shape[0], n_agents, n_agents, n_actions, n_actions), dtype=f.dtype, device=f.device)
    t = 0
    for i in range(n_agents):
        for j in range(n_agents - i - 1):
            g[:, i, i + j + 1, :, :] = g[:, i + j + 1, i, :, :] = _g[:, t, :, :]
            t += 1
    if not th.is_tensor(avail_actions):
        avail_actions = th.tensor(avail_actions)
    f[avail_actions == 0] = -9999999
    g[avail_actions.unsqueeze(1).unsqueeze(-2).repeat(1, n_agents, 1, n_actions, 1) == 0] = -9999999
    g[avail_actions.unsqueeze(2).unsqueeze(-1).repeat(1, 1, n_agents, 1, n_actions) == 0] = -9999999
    return f / n_agents, g / _g.shape[1]

class GreedyActionSelector:
    def __init__(self, args):
        self.greedy_lib = load_c_lib('./src/utils/greedy.cpp')
        self.alpha = args.leaky_alpha
        self.msg_iterations = args.msg_iterations
        self.onoff_configamount = args.onoff_configamount
        self.epsilon_init = args.epsilon_init
        self.epsilon_decay = args.epsilon_decay
        self.bav = args.best_action_version
    
    def solve(self, f, g, w_1, w_final, bias, avail_actions, device):
        f, g = f.detach(), g.detach()
        w_1, w_final, bias = w_1.detach(), w_final.detach(), bias.detach()
        f, g = preprocess_values(f, g, avail_actions)
        bs, n, m, l = f.shape[0], f.shape[1], f.shape[2], w_1.shape[2]

        _f = np.array(copy.deepcopy(f).cpu()).astype(ctypes.c_double)
        _g = np.array(copy.deepcopy(g).cpu()).astype(ctypes.c_double)
        _w_1 = np.array(copy.deepcopy(w_1).cpu()).astype(ctypes.c_double)
        _w_final = np.array(copy.deepcopy(w_final).cpu()).astype(ctypes.c_double)
        _bias = np.array(copy.deepcopy(bias).cpu()).astype(ctypes.c_double)
        _best_actions = np.zeros((bs, n, 1)).astype(ctypes.c_double)

        # print('new batch')
        # print(_f, _g, _w_1, w_final, _bias)

        self.greedy_lib.greedy(c_ptr(_f), c_ptr(_g), c_ptr(_best_actions), c_ptr(_w_1), c_ptr(_w_final), c_ptr(_bias), bs, n, m, l, c_double(self.alpha))

        best_actions = th.tensor(copy.deepcopy(_best_actions), dtype=th.int64, device=device)

        # best_actions =
        # print('actions = ', best_actions)

        return best_actions


        
    def maxsum_graph(self, f, g, avail_actions, device):
        f, g = f.detach(), g.detach()
        f, g = preprocess_values(f, g, avail_actions)
        bs, n, m = f.shape[0], f.shape[1], f.shape[2]

        _f = np.array(copy.deepcopy(f).cpu()).astype(ctypes.c_double)
        _g = np.array(copy.deepcopy(g).cpu()).astype(ctypes.c_double)
        _best_actions = np.zeros((bs, n, 1)).astype(ctypes.c_double)

        # print('new batch')
        # print(_f, _g, _w_1, w_final, _bias)

        self.greedy_lib.maxsum_graph(c_ptr(_f), c_ptr(_g), c_ptr(_best_actions), bs, n, m, self.msg_iterations)

        best_actions = th.tensor(copy.deepcopy(_best_actions), dtype=th.int64, device=device)

        # best_actions =
        # print('actions = ', best_actions)

        return best_actions
    def solve_graph(self, f, g, w_1, w_final, bias, avail_actions, device):
        f, g = f.detach(), g.detach()
        w_1, w_final, bias = w_1.detach(), w_final.detach(), bias.detach()
        f, g = preprocess_values(f, g, avail_actions)
        bs, n, m, l = f.shape[0], f.shape[1], f.shape[2], w_1.shape[2]

        _f = np.array(copy.deepcopy(f).cpu()).astype(ctypes.c_double)
        _g = np.array(copy.deepcopy(g).cpu()).astype(ctypes.c_double)
        _w_1 = np.array(copy.deepcopy(w_1).cpu()).astype(ctypes.c_double)
        _w_final = np.array(copy.deepcopy(w_final).cpu()).astype(ctypes.c_double)
        _bias = np.array(copy.deepcopy(bias).cpu()).astype(ctypes.c_double)
        _best_actions = np.zeros((bs, n, 1)).astype(ctypes.c_double)

        _wholeitert_timetotal = np.zeros((bs)).astype(ctypes.c_double)
        _maxsum_timetotal = np.zeros((bs)).astype(ctypes.c_double)
        _maxsum_iterrounds = np.zeros((bs)).astype(ctypes.c_double)

        # print('new batch')
        # print(_f, _g, _w_1, w_final, _bias)

        t0 = time.time()
        #self.greedy_lib.greedy_graph(c_ptr(_f), c_ptr(_g), c_ptr(_best_actions), c_ptr(_w_1), c_ptr(_w_final), c_ptr(_bias), bs, n, m, l, c_double(self.alpha), self.msg_iterations, self.onoff_configamount)
        self.greedy_lib.greedy_graph(c_ptr(_f), c_ptr(_g), c_ptr(_best_actions), c_ptr(_wholeitert_timetotal), c_ptr(_maxsum_timetotal), c_ptr(_maxsum_iterrounds), c_ptr(_w_1), c_ptr(_w_final), c_ptr(_bias), bs, n, m, l, c_double(self.alpha), self.msg_iterations, self.onoff_configamount, c_float(self.epsilon_init), c_float(self.epsilon_decay), self.bav)
        t1 = time.time()

        best_actions = th.tensor(copy.deepcopy(_best_actions), dtype=th.int64, device=device)

        # best_actions =
        # print('actions = ', best_actions)

        return [best_actions, t1-t0, _wholeitert_timetotal[0], _maxsum_timetotal[0], int(_maxsum_iterrounds[0])]
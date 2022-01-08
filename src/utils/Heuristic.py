import torch as th
import numpy as np
from utils.c_utils import load_c_lib, c_ptr, c_int, c_longlong, c_float, c_double
import copy
import ctypes

def preprocess_values(f, g, avail_actions):
    n_agents, n_actions = f.shape[1], f.shape[2]
    if not th.is_tensor(avail_actions):
        avail_actions = th.tensor(avail_actions)
    f[avail_actions == 0] = -9999999
    g[avail_actions.unsqueeze(1).unsqueeze(-2).repeat(1, n_agents, 1, n_actions, 1) == 0] = -9999999
    g[avail_actions.unsqueeze(2).unsqueeze(-1).repeat(1, 1, n_agents, 1, n_actions) == 0] = -9999999
    return f, g

class GreedyActionSelector:
    def __init__(self, args):
        self.greedy_lib = load_c_lib('./src/utils/greedy.cpp')
    
    def solve(self, f, g, w_1, w_final, leaky_alpha, avail_actions, device):
        f, g = f.detach(), g.detach()
        w_1, w_final = w_1.detach(), w_final.detach()
        f, g = preprocess_values(f, g, avail_actions)
        bs, n, m, l = f.shape[0], f.shape[1], f.shape[2], w_1.shape[2]

        _f = np.array(copy.deepcopy(f).cpu()).astype(ctypes.c_double)
        _g = np.array(copy.deepcopy(g).cpu()).astype(ctypes.c_double)
        _w_1 = np.array(copy.deepcopy(w_1).cpu()).astype(ctypes.c_double)
        _w_final = np.array(copy.deepcopy(w_final).cpu()).astype(ctypes.c_double)
        _best_actions = np.zeros((bs, n, 1)).astype(ctypes.c_double)

        print(_f, _g, _w_1, w_final)

        self.greedy_lib.greedy(c_ptr(_f), c_ptr(_g), c_ptr(_best_actions), c_ptr(_w_1), c_ptr(_w_final), bs, n, m, l, c_double(leaky_alpha))

        best_actions = th.tensor(copy.deepcopy(_best_actions), dtype=th.int64, device=device)

        # best_actions =
        print(best_actions)

        return best_actions


        
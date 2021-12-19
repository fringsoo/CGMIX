import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torh.nn.init
import numpy as np
from torch.nn.parameter import Parameter
import math


class QMixer_wos(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.in_dim = int((self.n_agents + self.n_agents ** 2) // 2)

        self.embed_dim = args.mixing_embed_dim

        self.w1 = Parameter(th.empty((self.in_dim, self.embed_dim)))
        th.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        self.b1 = Parameter(th.empty(self.embed_dim))
        if self.b1 is not None:
            fan_in, _ = th.nn.init._calculate_fan_in_and_fan_out(self.w1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            th.nn.init.uniform_(self.b1, -bound, bound)
        
        self.wfinal = Parameter(th.empty((self.embed_dim, 1)))
        th.nn.init.kaiming_uniform_(self.wfinal, a=math.sqrt(5))
        self.bfinal = Parameter(th.empty(1))
        if self.bfinal is not None:
            fan_in, _ = th.nn.init._calculate_fan_in_and_fan_out(self.wfinal)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            th.nn.init.uniform_(self.bfinal, -bound, bound)
        
        self.leakyrelu = nn.LeakyReLU(args.leaky_alpha)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.in_dim)
        # First layer
        w1 = th.abs(self.w1.unsqueeze(0).repeat(bs, 1, 1))
        b1 = self.b1.unsqueeze(0).repeat(bs, 1, 1)
        print(w1.shape, agent_qs.shape)
        hidden = self.leakyrelu(th.bmm(agent_qs, w1) + b1)# leakyReLU instead of ReLU
        # Second layer
        w_final = th.abs(self.wfinal.unsqueeze(0).repeat(bs, 1, 1))
        # State-dependent bias
        v = self.bfinal.unsqueeze(0).repeat(bs, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

    def get_para(self, states):
        bs = states.size(0)
        # states = states.reshape(-1, self.state_dim)
        # First layer
        w1 = th.abs(self.w1.unsqueeze(0).repeat(bs, 1, 1))
        b1 = self.b1.unsqueeze(0).repeat(bs, 1, 1)
        # Second layer
        w_final = th.abs(self.wfinal.unsqueeze(0).repeat(bs, 1, 1))
        w_1_detach = w1.detach()
        b_1_detach = b1.detach()
        w_final_detach = w_final.detach()
        return w_1_detach, b_1_detach, w_final_detach
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            #self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * (self.n_agents + self.n_agents ** 2) // 2)


            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * (self.n_agents + self.n_agents ** 2) // 2))
            
            # self.hyper_w_1_i = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
            #                                nn.ReLU(),
            #                                nn.Linear(hypernet_embed, self.embed_dim))

            # self.hyper_w_1_ij = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
            #                                nn.ReLU(),
            #                                nn.Linear(hypernet_embed, self.embed_dim))

            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        
        if self.args.mixingnetwork_weights == 'all1init_':
            for module in self.hyper_w_1.modules():
                if isinstance(module, nn.Linear):
                    th.nn.init.constant_(module.weight, 0)
                    th.nn.init.constant_(module.bias, 1)

            for module in self.hyper_w_final.modules():
                if isinstance(module, nn.Linear):
                    th.nn.init.constant_(module.weight, 0)
                    th.nn.init.constant_(module.bias, 1)
        

        # State dependent bias for hidden layer
        if self.args.mixingnetwork_bias in [0, 1]:
            self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        elif self.args.mixingnetwork_bias == 2:
            self.hyper_b_1 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                nn.ReLU(),
                                nn.Linear(self.embed_dim, self.embed_dim))
        elif self.args.mixingnetwork_bias == 10:
            self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
            for module in self.hyper_b_1.modules():
                if isinstance(module, nn.Linear):
                    th.nn.init.constant_(module.weight, 0)
                    th.nn.init.constant_(module.bias, 0)
        elif self.args.mixingnetwork_bias == 20:
            self.hyper_b_1 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                nn.ReLU(),
                                nn.Linear(self.embed_dim, self.embed_dim))
            for module in self.hyper_b_1.modules():
                if isinstance(module, nn.Linear):
                    th.nn.init.constant_(module.weight, 0)
                    th.nn.init.constant_(module.bias, 0)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        self.leakyrelu = nn.LeakyReLU(args.leaky_alpha)

    def con_w1i_w1ij(self, states):
        w_1_i = th.abs(self.hyper_w_1_i(states))
        w_1_i = w_1_i.repeat(1, self.n_agents).reshape([-1, self.n_agents, self.embed_dim])
        w_1_ij = th.abs(self.hyper_w_1_ij(states))
        w_1_ij = w_1_ij.repeat(1, (self.n_agents ** 2 - self.n_agents) // 2).reshape([-1, (self.n_agents ** 2 - self.n_agents) // 2, self.embed_dim])
        w_1 = th.cat([w_1_i, w_1_ij],dim=1)
        return w_1

    def forward(self, agent_qs, states):
        
        
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, (self.n_agents + self.n_agents ** 2) // 2)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        

        if self.args.mixingnetwork_weights == 'all1_':
            w1 = th.ones_like(w1)
        elif self.args.mixingnetwork_weights == 'stateless_':
            w1 = th.abs(self.hyper_w_1(th.zeros_like(states)))
        else:
            assert self.args.mixingnetwork_weights == 'nonneg_' or self.args.mixingnetwork_weights == 'all1init_'
        
        
        #w1 = th.abs(self.con_w1i_w1ij(states))

        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, (self.n_agents + self.n_agents ** 2) // 2, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        
    
        if self.args.mixingnetwork_bias in [1, 2, 10, 20]:
            hidden = self.leakyrelu(th.bmm(agent_qs, w1) + b1) # ReLU instead of elu
        elif self.args.mixingnetwork_bias == 0:
            hidden = self.leakyrelu(th.bmm(agent_qs, w1)) # ReLU instead of elu
        else:
            assert False
        
        #hidden = F.relu(th.bmm(agent_qs, w1) + b1) # ReLU instead of elu
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        if self.args.mixingnetwork_weights == 'all1_':
            w_final = th.ones_like(w_final)
        if self.args.mixingnetwork_weights == 'stateless_':
            w1 = th.abs(self.hyper_w_1(th.zeros_like(states))).view(-1, self.embed_dim, 1)
        else:
            assert self.args.mixingnetwork_weights == 'nonneg_'  or self.args.mixingnetwork_weights == 'all1init_'

        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
        '''
        #import pdb; pdb.set_trace()
        bs = agent_qs.size(0)
        v = self.V(states).view(bs, -1, 1)
        return th.sum(agent_qs, dim=2, keepdim=True) + v
        '''

    def get_para(self, states):
        #states = th.zeros_like(states) ###???
        
        states = states.reshape(-1, self.state_dim)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        
        if self.args.mixingnetwork_weights == 'all1_':
            w1 = th.ones_like(w1)
        if self.args.mixingnetwork_weights == 'stateless_':
            w1 = th.abs(self.hyper_w_1(th.zeros_like(states)))
        else:
            assert self.args.mixingnetwork_weights == 'nonneg_' or self.args.mixingnetwork_weights == 'all1init_'

        #w1 = th.abs(self.con_w1i_w1ij(states))

        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, (self.n_agents + self.n_agents ** 2) // 2, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        if self.args.mixingnetwork_weights == 'all1_':
            w_final = th.ones_like(w_final)
        if self.args.mixingnetwork_weights == 'stateless_':
            w1 = th.abs(self.hyper_w_1(th.zeros_like(states))).view(-1, self.embed_dim, 1)
        else:
            assert self.args.mixingnetwork_weights == 'nonneg_' or self.args.mixingnetwork_weights == 'all1init_'
        if self.args.mixingnetwork_bias == 0:
            b1 = th.zeros_like(b1)

        w_1_detach = w1.detach()
        b_1_detach = b1.detach()
        w_final_detach = w_final.detach()
        '''
        #import pdb; pdb.set_trace()
        bs = states.shape[0]
        w_1_detach = th.ones([bs, (self.n_agents + self.n_agents ** 2) // 2, self.embed_dim]).cuda()
        b_1_detach = th.zeros([bs, 1, self.embed_dim]).cuda()
        w_final_detach = th.ones([bs, self.embed_dim, 1]).cuda()
        '''
        #return w_1_detach, b_1_detach, w_final_detach
        return w_1_detach, w_final_detach, b_1_detach

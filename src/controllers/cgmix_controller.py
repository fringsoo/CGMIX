from pathlib import Path
from .basic_controller import BasicMAC
import torch as th
import torch.nn as nn
import numpy as np
import contextlib
import itertools
import torch_scatter
import copy
from math import factorial
from random import randrange
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_nostate import QMixer_wos
from utils.Heuristic import GreedyActionSelector


class CgmixMAC(BasicMAC):
    """ Multi-agent controller for a Deep Coordination Graph (DCG, Boehmer et al., 2020)"""

    # ================================ Constructors ===================================================================

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions

        #'''
        self.payoff_rank = args.cg_payoff_rank
        self.payoff_decomposition = isinstance(self.payoff_rank, int) and self.payoff_rank > 0
        #'''

        self.iterations = args.msg_iterations
        self.normalized = args.msg_normalized
        self.anytime = args.msg_anytime
        # Create neural networks for utilities and payoff functions
        self.utility_fun = self._mlp(self.args.rnn_hidden_dim, args.cg_utilities_hidden_dim, self.n_actions)
        payoff_out = self.n_actions ** 2
        
        #'''
        payoff_out = 2 * self.payoff_rank * self.n_actions if self.payoff_decomposition else self.n_actions ** 2
        self.payoff_out = payoff_out
        #'''
    
        self.payoff_fun = self._mlp(2 * self.args.rnn_hidden_dim, args.cg_payoffs_hidden_dim, payoff_out)
        # Create the edge information of the CG
        self.edges_from = None
        self.edges_to = None
        self.edges_n_in = None
        self._set_edges(self._edge_list())

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "qmix_wos":
                self.mixer = QMixer_wos(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
        self.leaky_alpha = args.leaky_alpha
        self.greedy_action_selector = GreedyActionSelector(args)

    # ================== DCG Core Methods =============================================================================

    def annotations(self, ep_batch, t, compute_grads=False, actions=None):
        """ Returns all outputs of the utility and payoff functions (Algorithm 1 in Boehmer et al., 2020). """
        with th.no_grad() if not compute_grads else contextlib.suppress():
            agent_inputs = self._build_inputs(ep_batch, t)
            self.hidden_states = self.agent(agent_inputs, self.hidden_states)[1].view(ep_batch.batch_size, self.n_agents, -1)
            f_i = self.utilities(self.hidden_states)
            f_ij = self.payoffs(self.hidden_states)
        return f_i, f_ij

    def utilities(self, hidden_states):
        """ Computes the utilities for a given batch of hidden states. """
        return self.utility_fun(hidden_states)

    def payoffs(self, hidden_states):
        """ Computes all payoffs for a given batch of hidden states. """
        # Construct the inputs for all edges' payoff functions and their flipped counterparts
        n = self.n_actions
        inputs = th.stack([th.cat([hidden_states[:, self.edges_from], hidden_states[:, self.edges_to]], dim=-1),
                           th.cat([hidden_states[:, self.edges_to], hidden_states[:, self.edges_from]], dim=-1)], dim=0)
        # Compute the payoff matrices for all edges (and flipped counterparts)
        output = self.payoff_fun(inputs)
        
        #output = th.zeros_like(output)
        


        
        if self.payoff_decomposition:
            # If the payoff matrix is decomposed, we need to de-decompose it here: ...
            dim = list(output.shape[:-1])
            # ... reshape output into left and right bases of the matrix, ...
            output = output.view(*[np.prod(dim) * self.payoff_rank, 2, n])
            # ... outer product between left and right bases, ...
            output = th.bmm(output[:, 0, :].unsqueeze(dim=-1), output[:, 1, :].unsqueeze(dim=-2))
            # ... and finally sum over the above outer products of payoff_rank base-pairs.
            output = output.view(*(dim + [self.payoff_rank, n, n])).sum(dim=-3)
        else:
            # Without decomposition, the payoff_fun output must only be reshaped

            output = output.view(*(list(output.shape[:-1]) + [n, n]))
        

        #output = output.view(*(list(output.shape[:-1]) + [n, n]))
        
        # The output of the backward messages must be transposed
        output[1] = output[1].transpose(dim0=-2, dim1=-1)
        # Compute the symmetric average of each edge with it's flipped counterpart
        return output.mean(dim=0)

    def q_values(self, f_i, f_ij, actions):
        """ Computes the Q-values for given utilities, payoffs and actions (Algorithm 2 in Boehmer et al., 2020). """
        n_batches = actions.shape[0]
        # Use the utilities for the chosen actions
        q_i = f_i.gather(dim=-1, index=actions).squeeze(dim=-1)
        q_i = q_i / self.n_agents
        # Use the payoffs for the chosen actions (if the CG contains edges)
        if len(self.edges_from) > 0:
            f_ij = f_ij.view(n_batches, len(self.edges_from), self.n_actions * self.n_actions)
            edge_actions = actions.gather(dim=-2, index=self.edges_from.view(1, -1, 1).expand(n_batches, -1, 1)) \
                * self.n_actions + actions.gather(dim=-2, index=self.edges_to.view(1, -1, 1).expand(n_batches, -1, 1))
            q_ij = f_ij.gather(dim=-1, index=edge_actions).squeeze(dim=-1)
            q_ij = q_ij / len(self.edges_from)
        # Return the Q-values for the given actions
        return q_i, q_ij

    def tot_values(self, f_i, f_ij, actions):
        q_i, q_ij = self.q_values(f_i, f_ij, actions)
        return q_i.sum(dim=-1) + q_ij.sum(dim=-1)

    def max_sum_C_graph(self, f_i, f_ij, available_actions=None):
        return self.greedy_action_selector.maxsum_graph(f_i, f_ij, avail_actions=available_actions, device=f_i.device)

    def max_sum(self, f_i, f_ij, available_actions=None):
        """ Finds the maximum Q-values and corresponding greedy actions for given utilities and payoffs.
            (Algorithm 3 in Boehmer et al., 2020)"""
        # All relevant tensors should be double to reduce accumulating precision loss
        in_f_i, f_i = f_i, f_i.double() / self.n_agents
        in_f_ij, f_ij = f_ij, f_ij.double() / len(self.edges_from)
        # Unavailable actions have a utility of -inf, which propagates throughout message passing
        if available_actions is not None:
            f_i = f_i.masked_fill(available_actions == 0, -float('inf'))
        # Initialize best seen value and actions for anytime-extension
        best_value = in_f_i.new_empty(f_i.shape[0]).fill_(-float('inf'))
        best_actions = f_i.new_empty(best_value.shape[0], self.n_agents, 1, dtype=th.int64, device=f_i.device)
        # Without edges (or iterations), CG would be the same as VDN: mean(f_i)
        utils = f_i
        ###
        # Perform message passing for self.iterations: [0] are messages to *edges_to*, [1] are messages to *edges_from*
        if len(self.edges_from) > 0 and self.iterations > 0:
            messages = f_i.new_zeros(2, f_i.shape[0], len(self.edges_from), self.n_actions)
            for iteration in range(self.iterations):
                # Recompute messages: joint utility for each edge: "sender Q-value"-"message from receiver"+payoffs/E
                joint0 = (utils[:, self.edges_from] - messages[1]).unsqueeze(dim=-1) + f_ij
                joint1 = (utils[:, self.edges_to] - messages[0]).unsqueeze(dim=-1) + f_ij.transpose(dim0=-2, dim1=-1)
                # Maximize the joint Q-value over the action of the sender
                messages[0] = joint0.max(dim=-2)[0]
                messages[1] = joint1.max(dim=-2)[0]
                # Normalization as in Kok and Vlassis (2006) and Wainwright et al. (2004)
                if self.normalized:
                    messages -= messages.mean(dim=-1, keepdim=True)
                # Create the current utilities of all agents, based on the messages
                msg = torch_scatter.scatter_add(src=messages[0], index=self.edges_to, dim=1, dim_size=self.n_agents)
                msg += torch_scatter.scatter_add(src=messages[1], index=self.edges_from, dim=1, dim_size=self.n_agents)
                utils = f_i + msg
                # Anytime extension (Kok and Vlassis, 2006)
                if self.anytime:
                    # Find currently best actions and the (true) value of these actions
                    actions = utils.max(dim=-1, keepdim=True)[1]
                    value = self.tot_values(in_f_i, in_f_ij, actions)
                    # Update best_actions only for the batches that have a higher value than best_value
                    change = value > best_value
                    best_value[change] = value[change]
                    best_actions[change] = actions[change]
        # Return the greedy actions and the corresponding message output averaged across agents
        if not self.anytime or len(self.edges_from) == 0 or self.iterations <= 0:
            _, best_actions = utils.max(dim=-1, keepdim=True)
        ###
        return best_actions

    def greedy(self, f_i, f_ij, w_1, w_final, bias, available_actions=None):
        #return self.greedy_action_selector.solve(f_i, f_ij, w_1, w_final, bias, avail_actions=available_actions, device=f_i.device)
        emb_dim = self.mixer.embed_dim
        w_1_i = w_1[:, :self.n_agents, :]
        w_1_ij = w_1[:, self.n_agents:, :]

        best_value = f_i.new_empty(f_i.shape[0]).fill_(-float('inf'))
        best_actions = f_i.new_empty(f_i.shape[0], self.n_agents, 1, dtype=th.int64, device=f_i.device)

        for iteration in range(0, 2 ** emb_dim):
            use_relu = np.zeros(emb_dim)
            for i in range(emb_dim):
                use_relu[i] = (iteration >> i) % 2
            #print(use_relu)
            k_i = th.zeros_like(w_1_i[:, :, 0])
            k_ij = th.zeros_like(w_1_ij[:, :, 0])
            res = th.zeros_like(best_value)
            for i in range(emb_dim):
                if use_relu[i] > 0.5:
                    k_i += w_1_i[:, :, i] * w_final[:, i]
                    k_ij += w_1_ij[:, :, i] * w_final[:, i]
                    res += (bias[:, 0, i] * w_final[: ,i, 0]) #b_1 or bias?
                else:
                    k_i += self.leaky_alpha * w_1_i[:, :, i] * w_final[:, i]
                    k_ij += self.leaky_alpha * w_1_ij[:, :, i] * w_final[:, i]
                    res += self.leaky_alpha * (bias[:, 0, i] * w_final[: ,i, 0])  #b_1 or bias?
            f_i_emb = f_i * k_i.unsqueeze(dim=-1)
            f_ij_emb = f_ij * k_ij.unsqueeze(dim=-1).unsqueeze(dim=-1)
            actions = self.max_sum(f_i_emb, f_ij_emb, available_actions)
            res += self.tot_values(f_i_emb, f_ij_emb, actions)
            change = res > best_value
            best_value[change] = res[change]
            best_actions[change] = actions[change]

        return best_actions










    def greedy_heuristic(self, f_i, f_ij, w_1, w_final, bias, available_actions=None):
        emb_dim = self.mixer.embed_dim
        w_1_i = w_1[:, :self.n_agents, :]
        w_1_ij = w_1[:, self.n_agents:, :]

        best_value = f_i.new_empty(f_i.shape[0]).fill_(-float('inf'))
        best_actions = f_i.new_empty(f_i.shape[0], self.n_agents, 1, dtype=th.int64, device=f_i.device)

        bs = f_i.shape[0]
        explored = np.zeros([bs, 2 ** emb_dim], dtype=np.int)
        #explored = [0 for i in range(2 ** emb_dim)]
        use_relu = np.ones([bs, emb_dim], dtype=np.int)
        #use_relu = np.ones(emb_dim, dtype=np.int)
        moving_tag = self.on_off_tag(use_relu)
        
        for iteration in range(10):
            tags = self.on_off_tag(use_relu)
            explored[range(bs), tags] = 1
                        
            k_i = th.zeros_like(w_1_i[:, :, 0])
            k_ij = th.zeros_like(w_1_ij[:, :, 0])
            res = th.zeros_like(best_value)
            
            for i in range(emb_dim):
                slope = th.from_numpy((use_relu[:, i] - 1) * (1 - self.leaky_alpha) + 1).to(f_i.device)#???
                
                k_i += slope.unsqueeze(dim=-1)*(w_1_i[:, :, i] * w_final[:, i])
                k_ij += slope.unsqueeze(dim=-1)*(w_1_ij[:, :, i] * w_final[:, i])
                res += slope*(bias[:, 0, i] * w_final[: ,i, 0])
            
            f_i_emb = f_i * k_i.unsqueeze(dim=-1)
            f_ij_emb = f_ij * k_ij.unsqueeze(dim=-1).unsqueeze(dim=-1)
            actions = self.max_sum(f_i_emb, f_ij_emb, available_actions)
            res += self.tot_values(f_i_emb, f_ij_emb, actions)
            change = res > best_value
            best_value[change] = res[change]
            best_actions[change] = actions[change]

            # print(w_1_i[0][1][1], w_1_ij[0][2][2], w_final[0][0])
            # print(k_i, k_ij, res)
            # print(use_relu, int(actions[0][0]), int(actions[0][1]), int(actions[0][2]), int(actions[0][3]), res)
            
            node = th.zeros([bs, emb_dim])
            node_i = f_i.unsqueeze(dim=-1) * w_1_i.unsqueeze(dim=-2)
            node_ij = f_ij.unsqueeze(dim=-1) * w_1_ij.unsqueeze(dim=-2).unsqueeze(dim=-2)
            
            for i in range(emb_dim):
                node[:, i] = self.tot_values(node_i[:,:,:,i], node_ij[:,:,:,:,i], actions) + bias[:, 0, i]
        
            real_use_relu = np.array(node > 0).astype(int)
            
            different_real = self.on_off_tag(use_relu) != self.on_off_tag(real_use_relu)
            real_explored = explored[range(bs), self.on_off_tag(real_use_relu)]
            
            for b in range(bs):
                if different_real[b] and (not real_explored[b]):
                    use_relu[b] = real_use_relu[b]
                else:
                    while explored[b][moving_tag[b]] and moving_tag[b] > 0:
                        moving_tag[b] -= 1
                    use_relu[b] = self.tag_on_off(moving_tag[b])

        return best_actions

    def greedy_heuristic_C(self, f_i, f_ij, w_1, w_final, bias, available_actions=None):
        return self.greedy_action_selector.solve_graph(f_i, f_ij, w_1, w_final, bias, avail_actions=available_actions, device=f_i.device)


    def on_off_tag(self, use_relu):
        base = np.array([2**i for i in range(self.mixer.embed_dim)])
        tags = use_relu.dot(base.transpose())
        return tags

    def tag_on_off(self, tag):
        use_relu = np.zeros(self.mixer.embed_dim, dtype=np.int)
        for i in range(self.mixer.embed_dim):
            use_relu[i] = (tag >> i) % 2
        return use_relu



    # ================== Override methods of BasicMAC to integrate DCG into PyMARL ====================================

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        state = ep_batch["state"][:, t_ep]
        f_i, f_ij = self.annotations(ep_batch, t_ep)
        
        assert f_i.shape[0] == 1
        
        w_1, w_final, bias = self.mixer.get_para(state)
        
        #import time
        #t0 = time.time()
        #actions1 = self.max_sum(f_i, f_ij, avail_actions)
        #t1 = time.time()
        #actions2 = self.max_sum_C_graph(f_i, f_ij, avail_actions)
        #t2 = time.time()
        #assert th.all(actions1==actions2)
        #actions = actions2

        #actions_traverse = self.greedy(f_i, f_ij, w_1, w_final, bias, avail_actions)
        #t3 = time.time()
        #actions_heuristic = self.greedy_heuristic(f_i, f_ij, w_1, w_final, bias, avail_actions)
        #t4 = time.time()
        actions_heuristic_C = self.greedy_heuristic_C(f_i, f_ij, w_1, w_final, bias, avail_actions)
        #t5 = time.time()
        
        actions = actions_heuristic_C
        #actions = actions_traverse

        # if not th.all(actions_traverse == actions_heuristic_C):
        #     print(actions_traverse.squeeze().transpose(-1,0), actions_heuristic_C.squeeze().transpose(-1,0))
        #     print((actions_traverse == actions_heuristic_C).squeeze().transpose(-1,0))
        # if not th.all(actions_heuristic == actions_heuristic_C):
        #     print(actions_heuristic.squeeze().transpose(-1,0), actions_heuristic_C.squeeze().transpose(-1,0))
        #     print((actions_heuristic == actions_heuristic_C).squeeze().transpose(-1,0))
        # if not th.all(actions_traverse == actions_heuristic):
        #     print(actions_traverse.squeeze().transpose(-1,0), actions_heuristic.squeeze().transpose(-1,0))
        #     print((actions_traverse == actions_heuristic).squeeze().transpose(-1,0))
        
        #print(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4)
        
        policy = f_i.new_zeros(ep_batch.batch_size, self.n_agents, self.n_actions)
        policy.scatter_(dim=-1, index=actions, src=policy.new_ones(1, 1, 1).expand_as(actions))
        chosen_actions = self.action_selector.select_action(policy[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, actions=None, w_1 = None, w_final = None, test_mode=False):
        if actions is not None:
            f_i, f_ij = self.annotations(ep_batch, t, compute_grads=True, actions=actions)
            
            #print('foward with action', f_i.shape[0])

            q_i, q_ij = self.q_values(f_i, f_ij, actions)
            return q_i, q_ij
        else:
            f_i, f_ij = self.annotations(ep_batch, t)

            #print('foward with no action', f_i.shape[0])

            if w_1 is None:
                state = ep_batch["state"][:, t]
                w_1, w_final, bias = self.mixer.get_para(state)
            #actions = self.greedy(f_i, f_ij, w_1, w_final, bias, available_actions=ep_batch['avail_actions'][:, t])
            
            avail_actions = ep_batch['avail_actions'][:, t]
            #import time
            #t0 = time.time()
            #actions1 = self.max_sum(f_i, f_ij, avail_actions)
            #t1 = time.time()
            #actions2 = self.max_sum_C_graph(f_i, f_ij, avail_actions)
            #t2 = time.time()
            #assert th.all(actions1==actions2)
            # action = actions2
        
            #actions_traverse = self.greedy(f_i, f_ij, w_1, w_final, bias, available_actions=avail_actions)
            #t3 = time.time()
            #actions_heuristic = self.greedy_heuristic(f_i, f_ij, w_1, w_final, bias, available_actions=avail_actions)
            #t4 = time.time()
            actions_heuristic_C = self.greedy_heuristic_C(f_i, f_ij, w_1, w_final, bias, available_actions=avail_actions)
            #t5 = time.time()

            actions = actions_heuristic_C
            #actions = actions_traverse

            # if not th.all(actions_traverse == actions_heuristic_C):
            #     print(actions_traverse.squeeze().transpose(-1,0), actions_heuristic_C.squeeze().transpose(-1,0))
            #     print((actions_traverse == actions_heuristic_C).squeeze().transpose(-1,0))
            # if not th.all(actions_heuristic == actions_heuristic_C):
            #     print(actions_heuristic.squeeze().transpose(-1,0), actions_heuristic_C.squeeze().transpose(-1,0))
            #     print((actions_heuristic == actions_heuristic_C).squeeze().transpose(-1,0))
            # if not th.all(actions_traverse == actions_heuristic):
            #     print(actions_traverse.squeeze().transpose(-1,0), actions_heuristic.squeeze().transpose(-1,0))
            #     print((actions_traverse == actions_heuristic).squeeze().transpose(-1,0))
            
            #print(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4)
            return actions

    def cuda(self):
        """ Moves this controller to the GPU, if one exists. """
        self.agent.cuda()
        self.utility_fun.cuda()
        self.payoff_fun.cuda()
        if self.edges_from is not None:
            self.edges_from = self.edges_from.cuda()
            self.edges_to = self.edges_to.cuda()
            self.edges_n_in = self.edges_n_in.cuda()
        if self.mixer is not None:
            self.mixer.cuda()

    def parameters(self):
        """ Returns a generator for all parameters of the controller. """
        if self.mixer is not None:
            param = itertools.chain(BasicMAC.parameters(self), self.utility_fun.parameters(), self.payoff_fun.parameters(), self.mixer.parameters())
        else:
            param = itertools.chain(BasicMAC.parameters(self), self.utility_fun.parameters(), self.payoff_fun.parameters())
        return param

    def load_state(self, other_mac):
        """ Overwrites the parameters with those from other_mac. """
        BasicMAC.load_state(self, other_mac)
        self.utility_fun.load_state_dict(other_mac.utility_fun.state_dict())
        self.payoff_fun.load_state_dict(other_mac.payoff_fun.state_dict())
        if self.mixer is not None:
            self.mixer.load_state_dict(other_mac.mixer.state_dict())

    def save_models(self, path):
        """ Saves parameters to the disc. """
        BasicMAC.save_models(self, path)
        th.save(self.utility_fun.state_dict(), "{}/utilities.th".format(path))
        th.save(self.payoff_fun.state_dict(), "{}/payoffs.th".format(path))
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))

    def load_models(self, path):
        """ Loads parameters from the disc. """
        BasicMAC.load_models(self, path)
        self.utility_fun.load_state_dict(th.load("{}/utilities.th".format(path), map_location=lambda storage, loc: storage))
        self.payoff_fun.load_state_dict(th.load("{}/payoffs.th".format(path), map_location=lambda storage, loc: storage))
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))

    # ================== Private methods to help the constructor ======================================================

    @staticmethod
    def _mlp(input, hidden_dims, output):
        """ Creates an MLP with the specified input and output dimensions and (optional) hidden layers. """
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d
        layers.append(nn.Linear(dim, output))
        return nn.Sequential(*layers)

    def _edge_list(self):
        """ Specifies edges for various topologies. """
        edges = []
        edges = [[(j, i + j + 1) for i in range(self.n_agents - j - 1)] for j in range(self.n_agents - 1)]
        edges = [e for l in edges for e in l]
        return edges

    def _set_edges(self, edge_list):
        """ Takes a list of tuples [0..n_agents)^2 and constructs the internal CG edge representation. """
        self.edges_from = th.zeros(len(edge_list), dtype=th.long)
        self.edges_to = th.zeros(len(edge_list), dtype=th.long)
        for i, edge in enumerate(edge_list):
            self.edges_from[i] = edge[0]
            self.edges_to[i] = edge[1]
        self.edges_n_in = torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                    index=self.edges_to, dim=0, dim_size=self.n_agents) \
                          + torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                      index=self.edges_from, dim=0, dim_size=self.n_agents)
        self.edges_n_in = self.edges_n_in.float()

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


class CgmixMAC(BasicMAC):
    """ Multi-agent controller for a Deep Coordination Graph (DCG, Boehmer et al., 2020)"""

    # ================================ Constructors ===================================================================

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        self.iterations = args.msg_iterations
        self.normalized = args.msg_normalized
        self.anytime = args.msg_anytime
        # Create neural networks for utilities and payoff functions
        self.utility_fun = self._mlp(self.args.rnn_hidden_dim, args.cg_utilities_hidden_dim, self.n_actions)
        payoff_out = self.n_actions ** 2
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
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

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
        output = output.view(*(list(output.shape[:-1]) + [n, n]))
        # The output of the backward messages must be transposed
        output[1] = output[1].transpose(dim0=-2, dim1=-1)
        # Compute the symmetric average of each edge with it's flipped counterpart
        return output.mean(dim=0)

    def q_values(self, f_i, f_ij, actions):
        """ Computes the Q-values for given utilities, payoffs and actions (Algorithm 2 in Boehmer et al., 2020). """
        n_batches = actions.shape[0]
        # Use the utilities for the chosen actions
        q_i = f_i.gather(dim=-1, index=actions).squeeze(dim=-1)
        # Use the payoffs for the chosen actions (if the CG contains edges)
        if len(self.edges_from) > 0:
            f_ij = f_ij.view(n_batches, len(self.edges_from), self.n_actions * self.n_actions)
            edge_actions = actions.gather(dim=-2, index=self.edges_from.view(1, -1, 1).expand(n_batches, -1, 1)) \
                * self.n_actions + actions.gather(dim=-2, index=self.edges_to.view(1, -1, 1).expand(n_batches, -1, 1))
            q_ij = f_ij.gather(dim=-1, index=edge_actions).squeeze(dim=-1)
        # Return the Q-values for the given actions
        return q_i, q_ij

    def greedy(self, f_i, f_ij, available_actions=None):
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
                    value = self.q_values(in_f_i, in_f_ij, actions)
                    # Update best_actions only for the batches that have a higher value than best_value
                    change = value > best_value
                    best_value[change] = value[change]
                    best_actions[change] = actions[change]
        # Return the greedy actions and the corresponding message output averaged across agents
        if not self.anytime or len(self.edges_from) == 0 or self.iterations <= 0:
            _, best_actions = utils.max(dim=-1, keepdim=True)
        return best_actions

    # ================== Override methods of BasicMAC to integrate DCG into PyMARL ====================================

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        f_i, f_ij = self.annotations(ep_batch, t_ep)
        chosen_actions = self.action_selector.select_action(f_i[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, actions=None, test_mode=False):
        if actions is not None:
            f_i, f_ij = self.annotations(ep_batch, t, compute_grads=True, actions=actions)
            q_i, q_ij = self.q_values(f_i, f_ij, actions)
            return q_i, q_ij
        else:
            f_i, f_ij = self.annotations(ep_batch, t)
            _, actions = (th.zeros((f_i.size(0), f_i.size(1), 1))).max(dim=-1, keepdim=True)
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
            self.mixer.laod_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))

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

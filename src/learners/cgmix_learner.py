from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
import torch as th


class CgmixLearner(QLearner):
    """ QLearner for a Deep Coordination Graph (DCG, Boehmer et al., 2020). """

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """ Overrides the train method from QLearner. """

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Calculate estimated Q-Values
        mac_f_i = []
        mac_f_ij = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            f_i, f_ij = self.mac.forward(batch, t=t, actions=actions[:, t])
            mac_f_i.append(f_i)
            mac_f_ij.append(f_ij)
        mac_f_i = th.stack(mac_f_i, dim=1)
        mac_f_ij = th.stack(mac_f_ij, dim=1)

        target_f_i = []
        target_f_ij = []
        self.target_mac.init_hidden(batch.batch_size)
        w_1, w_final = self.target_mixer.get_w(batch["state"][:, 1:]) # Should I use target_mixer, state[1:] ??
        for t in range(batch.max_seq_length):
            greedy = self.mac.forward(batch, t=t, actions=None, w_1=w_1, w_final=w_final)
            f_i, f_ij = self.target_mac.forward(batch, t=t, actions=greedy)
            target_f_i.append(f_i)
            target_f_ij.append(f_ij)
        target_f_i = th.stack(target_f_i, dim=1)
        target_f_ij = th.stack(target_f_ij, dim=1)
        
        mac_out = th.cat((mac_f_i, mac_f_ij), dim=2)
        target_out = th.cat((target_f_i, target_f_ij), dim=2)

        

        # Mix
        if self.mixer is not None:
            chosen_action_qval = self.mixer(mac_out, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_out, batch["state"][:, 1:])

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Calculate TD-error and masked loss for 1-step Q-Learning targets
        td_error = (chosen_action_qval - targets.detach())
        mask = mask.expand_as(td_error)
        td_error = td_error * mask
        loss = (td_error ** 2).sum() / mask.sum()

        # Optimise the loss
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # Update target network if it is time
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Log important learning variables
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (mac_out * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

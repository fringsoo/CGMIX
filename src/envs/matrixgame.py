from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import numpy as np
import random


class MatrixGameEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
            self,
            n_agents=4,
            n_actions=2,
            episode_limit=2,
            seed=None
    ):
        self._seed = seed
        if seed == None:
            self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)

        # Map arguments
        self.n_agents = n_agents

        # Actions
        self.n_actions = n_actions

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.episode_limit = episode_limit

        self.state = 0
        self.reward1 = 7
        self.reward2 = [0, -.1, .3, .1, 8]

    # def step(self, actions, g, graph, test_mode):
    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        reward = 0
        terminated = False

        if self.state == 0:
            if actions[0] == 0:
                self.state = 1
            else:
                self.state = 2
        elif self.state == 1:
            reward = self.reward1
        else:
            cnt = 0
            for action in actions:
                if action == 1:
                    cnt += 1
            reward = self.reward2[cnt]
            


        info['battle_won'] = False

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1


        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        state = np.zeros(3)
        state[self.state] = 1
        return state

    def get_obs_size(self):
        """Returns the size of the observation."""
        return 3

    def get_state(self):
        """Returns the global state."""
        state = np.zeros(3)
        state[self.state] = 1
        return state

    def get_state_size(self):
        """Returns the size of the global state."""
        return 3

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        self.state = 0

        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game
        }
        return stats
## TODO:
## - Build multi-agent adapter

import gym
from gym import spaces
import numpy as np
import random
import cv2
from collections import namedtuple

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland_training.src.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland_training.src import tree_observationpytjh

x_dim = 36
y_dim = 36

# Custom observation builder

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

# Use a the malfunction generator to break agents from time to time
stochastic_data = MalfunctionParameters(malfunction_rate=1/10000,  # Rate of malfunction occurence
                                        min_duration=15,  # Minimal duration of malfunction
                                        max_duration=50  # Max duration of malfunction
                                        )

random.seed(41)
np.random.seed(41)

class FooEnv(gym.Env):
    def __init__(self, n_cars=3 , n_acts=5, min_obs=-1, max_obs=1, n_nodes=2, n_feats=11, ob_radius=10):

        self.tree_obs = tree_observation.TreeObservation(n_nodes)
        self.total_feats = n_feats * sum([4**i for i in range(n_nodes+1)])
        # self.action_space = spaces.MultiDiscrete([n_acts]*n_cars)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, shape=(self.total_feats,), dtype=np.float32)
        self.n_cars = n_cars
        self.n_nodes = n_nodes
        self.ob_radius = ob_radius
        

        self._rail_env = RailEnv(
            width=x_dim,
            height=y_dim,
            rail_generator=sparse_rail_generator(max_num_cities=3,
                                                # Number of cities in map (where train stations are)
                                                seed=123,  # Random seed
                                                grid_mode=False,
                                                max_rails_between_cities=2,
                                                max_rails_in_city=3),
            schedule_generator=sparse_schedule_generator(speed_ration_map),
            number_of_agents=n_cars,
            malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
            obs_builder_object=self.tree_obs)
        self.renderer = RenderTool(self._rail_env, gl="PILSVG")

        self.action_dict = dict()
        self.info = dict()
        self.updates = dict()
        self.old_obs = dict()
        self.cum_reward = dict()

    # @property
    def states(self):
        return self.observation_space

    # @property
    def actions(self):
        return self.action_space

    def max_episode_timesteps(self):
        return 10000


    def step(self, action):
        """
        ----------
        action : int
        return obs, reward, resetFlag, info
            see https://gym.openai.com/docs/#observations
        """
        # print(action)

        # agent_id, action = action



        for agent_id in range(self._rail_env.get_num_agents()):
            if self.info['action_required'][agent_id]:
                self.updates[agent_id] = True
            else:
                self.updates[agent_id] = False
                action[agent_id] = 0
            self.action_dict.update({agent_id: action[agent_id]})

        next_obs, all_rewards, done, self.info = self._rail_env.step(self.action_dict)


        for agent_id in range(self._rail_env.get_num_agents()):
            # Check if agent is finished
            if not done[agent_id]:
                if agent_id in self.cum_reward.keys():
                    self.cum_reward.update({agent_id: self.cum_reward[agent_id] - 0.1})
                else:
                    self.cum_reward.update({agent_id: -0.1})

                next_obs[agent_id] = normalize_observation(next_obs[agent_id], self.n_nodes, self.ob_radius)
            else:
                next_obs[agent_id] = self.old_obs[agent_id]
            self.old_obs[agent_id] = next_obs[agent_id].copy()

        return next_obs, all_rewards, done, {}


    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        return obs: initial observation of the space
        """
        self.action_dict = dict()
        self.info = dict()
        self.updates = dict()
        self.old_obs = dict()
        self.cum_reward = dict()


        obs, self.info = self._rail_env.reset(True, True)
        for agent_id in range(self._rail_env.get_num_agents()):
            if obs[agent_id]:
                obs[agent_id] = normalize_observation(obs[agent_id], self.n_nodes, self.ob_radius)
        feats = [f.reshape(1,-1) for f in obs.values()]
        obs = np.concatenate(feats)
        self.renderer.reset()
        
        return obs

    def render(self, mode=None):
        
        self.renderer.render_env()
        image = self.renderer.get_image()
        cv2.imshow('Render', image)
        cv2.waitKey(20)

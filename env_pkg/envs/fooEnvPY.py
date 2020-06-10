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
from env_pkg.envs.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters


x_dim = 36
y_dim = 36

# Custom observation builder
TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))
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

random.seed(1)
np.random.seed(1)

class FooEnv(gym.Env):
    def __init__(self, n_cars=1, n_acts=5, min_obs=-1, max_obs=1, n_nodes=2, n_feats=11, ob_radius=10):

        self.action_space = spaces.Tuple([spaces.Discrete(n_acts)]*n_cars) 
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, shape=(n_cars, n_nodes*n_feats*ob_radius), dtype=np.float32)
        self.n_cars = n_cars
        self.n_nodes = n_nodes
        self.ob_radius = ob_radius
        self.total_feats = n_nodes*n_feats*ob_radius

        self._rail_env = RailEnv(
            width=x_dim,
            height=y_dim,
            rail_generator=sparse_rail_generator(max_num_cities=3,
                                                # Number of cities in map (where train stations are)
                                                seed=1,  # Random seed
                                                grid_mode=False,
                                                max_rails_between_cities=2,
                                                max_rails_in_city=3),
            schedule_generator=sparse_schedule_generator(speed_ration_map),
            number_of_agents=n_cars,
            malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
            obs_builder_object=TreeObservation)

        self.action_dict = dict()
        self.info = dict()


    def step(self, action):
        """
        ----------
        action : int
        return obs, reward, resetFlag, info
            see https://gym.openai.com/docs/#observations
        """

        # Agent action + observation
        self.action_dict.update({0: action})
        next_obs, all_rewards, done, self.info = self._rail_env.step(self.action_dict)

        # Check if agent is finished
        if done[0]:
            print(done)
            # FIXME: This is probably a stupid way to return the final observation, but keras rl seems to expect one
            return np.zeros((total_feats)), all_rewards[0], True, {}

        # Only normalise observation if we're not done yet 
        else:
            next_obs = normalize_observation(next_obs[0], self.n_nodes, self.ob_radius)
            return next_obs, all_rewards[0], done[0], {}


    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        return obs: initial observation of the space
        """
        obs, self.info = self._rail_env.reset(True, True)
        obs = normalize_observation(obs[0], self.n_nodes, self.ob_radius)
        return obs

    def render(self, mode):
        env_renderer = RenderTool(self._rail_env, gl="PILSVG")
        env_renderer.render_env()
        image = env_renderer.get_image()
        cv2.imshow('Render', image)
        cv2.waitKey(1)
